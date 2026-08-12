import fitz

doc = fitz.open("data/Acute rheumatic fever.pdf")
text = ""
for page in doc:
    text += page.get_text()
    
##print(text)
