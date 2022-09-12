from paper_parser.parser import PaperParser


def gen_pdf_form():
    pp = PaperParser('/mnt/nvme-data1/bhanu/code-bases/papertoweb/samples/test.jpg','base')
    pp.make_editable_pdf('/mnt/nvme-data1/bhanu/code-bases/papertoweb/outputs/sample_form.pdf')


if __name__=="__main__":
    gen_pdf_form()