import fitz

doc = fitz.open("data/First-Aid-for-the-USMLE-Step2-2025.txt")
text = ""
for page in doc:
    text += page.get_text()
    
print("extracted!")
