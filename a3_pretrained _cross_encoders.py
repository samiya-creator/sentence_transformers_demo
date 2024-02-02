from sentence_transformers import CrossEncoder

model = CrossEncoder("model_name", max_length=512)
scores = model.predict([("Query1", "Paragraph1"), ("Query1", "Paragraph2")])

# For Example
scores = model.predict([
    ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
])