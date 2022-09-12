from paper_parser.parser import PaperParser


def test_pdfform():
    pp = PaperParser('/mnt/nvme-data1/bhanu/code-bases/papertoweb/samples/test.jpg','base')
    pp.make_editable_pdf('/mnt/nvme-data1/bhanu/code-bases/papertoweb/outputs/sample_form.pdf')

