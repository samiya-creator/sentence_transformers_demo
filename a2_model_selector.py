from sentence_transformers import SentenceTransformer, util
def get_model_data(model_name):
    print(f"Getting data for model: {model_name}")
    
    model = SentenceTransformer(model_name)

    query_embedding = model.encode("How big is London")
    passage_embedding = model.encode([
        "London has 9,787,426 inhabitants at the 2011 census",
        "London is known for its finacial district",
    ])

    print("Similarity:", util.dot_score(query_embedding, passage_embedding))


# def choose_model():
#     model_list = [
#         "multi-qa-MiniLM-L6-dot-v1",
#         "multi-qa-distilbert-dot-v1",
#         "multi-qa-mpnet-base-dot-v1"
#     ]
#     basic_info = """
# Model Names:
#     1. multi-qa-MiniLM-L6-dot-v1
#     2. multi-qa-distilbert-dot-v1
#     3. multi-qa-mpnet-base-dot-v1
# 
# Select Model Number: """
# 
#     print(basic_info)
#     modelid = int(input())
#     
#     match modelid:
#         case 1:
#             get_model_data("multi-qa-MiniLM-L6-dot-v1")
#         case 2:
#             get_model_data("multi-qa-distilbert-dot-v1")
#         case 3:
#             get_model_data("multi-qa-mpnet-base-dot-v1")

def choose_model():
    models = {
        1: "multi-qa-MiniLM-L6-dot-v1",
        2: "multi-qa-distilbert-dot-v1",
        3: "multi-qa-mpnet-base-dot-v1"
    }

    def print_model_names():
        print("Model Names:")
        for index, model_name in models.items():
            print(f"{index}. {model_name}")

    print_model_names()

    while True:
        print("Select a model number or enter '0' to exit:")
        try:
            user_input = input()
            if user_input == '0':
                print("Exiting the program.")
                break
            model_id = int(user_input)
            if model_id in models:
                get_model_data(models[model_id])
            else:
                print("Invalid model number. Please select a valid model.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            
if __name__ == "__main__":
    choose_model()