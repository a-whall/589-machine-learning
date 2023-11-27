import PyPDF2

input_pdf = PyPDF2.PdfReader("589 HW7-1.pdf")
output_pdf = PyPDF2.PdfWriter()

for i in range(3):
    output_pdf.add_page(input_pdf.pages[i])

with open("newfile.pdf", "wb") as output_stream:
    output_pdf.write(output_stream)