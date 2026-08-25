import calamancy

nlp = calamancy.load("tl_calamancy_md-0.1.0")
doc = nlp("Ang ganda ng product, super satisfied ako!")

for token in doc:
    print(token.text, token.pos_, token.dep_)