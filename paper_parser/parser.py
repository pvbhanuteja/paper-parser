import warnings
import logging
import requests



from datasets import load_dataset 
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoProcessor

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.colors import magenta, pink, blue, green
import easyocr

logger = logging.getLogger(__name__)



class PaperParser(object):

    def __init__(self, img_path, model_type='base', width_ths=0.25):
        """ Initialize the client. Do this before using other Label Studio SDK classes and methods in your script.
        Parameters
        ----------
        img_path: Path
            Path to image
        model_type: str
            choose between 'base' or 'large'
        width_ths = float
            Refer https://www.jaided.ai/easyocr/documentation/ to ser width_ths default=0.25
            width_ths is thres to merge easyocr's bboxes 
        """
        self.processor = AutoProcessor.from_pretrained(f"microsoft/layoutlmv3-{model_type}",
                                        apply_ocr=False)

        self.model = AutoModelForTokenClassification.from_pretrained(f"pvbhanuteja/llmv3-{model_type}-funsd")
        self.image = Image.open(img_path) 
        self.reader = easyocr.Reader(['en'])
        self.bounds = self.reader.readtext(np.array(self.image),width_ths = 0.25)
        self.width, self.height = self.image.size
        self.words = []
        self.boxes = []
        for bound in self.bounds:
            p0, p1, p2, p3 = bound[0]
            a = min(p0[0],p3[0])
            b = min(p0[1],p1[1])
            c = max(p1[0],p2[0])
            d = max(p2[1],p3[1])
            self.boxes.append([int(a/self.width * 1000),int(b/self.height * 1000) , int(c/self.width * 1000),int(d/self.height * 1000)])
            self.words.append(bound[1])
        self.encoding = self.processor(self.image, self.words,boxes=self.boxes, return_offsets_mapping=True, return_tensors="pt")
        self.offset_mapping = self.encoding.pop('offset_mapping')
        with torch.no_grad():
            self.outputs = self.model(**self.encoding)
        # labels = encoding.labels.squeeze().tolist()
        self.predictions = self.outputs.logits.argmax(-1).squeeze().tolist()
        self.token_boxes = self.encoding.bbox.squeeze().tolist()
        self.is_subword = np.array(self.offset_mapping.squeeze().tolist())[:,0] != 0
        self.true_predictions = [self.model.config.id2label[pred] for idx, pred in enumerate(self.predictions) if not self.is_subword[idx]]
        self.true_boxes = [self.unnormalize_box(box, self.width, self.height) for idx, box in enumerate(self.token_boxes) if not self.is_subword[idx]]
        
    def unnormalize_box(self,bbox, width, height):
            return[
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),]

    def make_editable_pdf(self,pdf_save_path):
        c = canvas.Canvas(pdf_save_path,pagesize=(self.width,self.height))
        form = c.acroForm
        cn=0
        for prediction, box in zip(self.true_predictions[1:-1], self.true_boxes[1:-1]):
            predicted_label = prediction[2:].lower()
            if predicted_label != 'answer':
                c.drawString(box[0], self.height - box[1],self.words[cn])
                cn+=1
            else:
                form.textfield(name=self.words[cn],
                   x=box[0], y=self.height - box[1], borderStyle='inset',
                   width= box[2], forceBorder=True)
                cn+=1
        c.save()
        return('pdf saved')
        
if __name__ =='__main__':
    pp = PaperParser('/mnt/nvme-data1/bhanu/code-bases/papertoweb/samples/test.jpg','base')
    pp.make_editable_pdf('/mnt/nvme-data1/bhanu/code-bases/papertoweb/outputs/sample_form.pdf')