import pandas as pd
import os
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from openai import OpenAI
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_examples_with_gpt(prompt, num_examples=10):
    client = OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate {num_examples} variations of the following example: '{prompt}'"}
        ],
        max_tokens=100,
        temperature=0.7
    )
    examples = [choice.message.content.strip() for choice in completion.choices]
    return examples

def generate_synthetic_data():
    intents = {
        "transactions": [
            "How do I send Bitcoin to another wallet?",
            "What is the process to receive Ethereum?",
            "How can I track my cryptocurrency transactions?",
            "How do I transfer crypto to my exchange account?",
            "What fees are associated with sending Litecoin?",
            "How do I check the status of a crypto transaction?",
            "Can I reverse a Bitcoin transaction?"
        ],
        "yield_farming": [
            "What are the best yield farming strategies?",
            "How can I earn passive income through yield farming?",
            "What is yield farming in DeFi?",
            "Which platforms offer the highest yield farming returns?",
            "What risks are associated with yield farming?",
            "How do I start with yield farming?",
            "What is the difference between yield farming and staking?"
        ],
        "token_info": [
            "Where can I find information about the latest tokens?",
            "How do I research a new cryptocurrency token?",
            "What is the market cap of a specific token?",
            "What are the top performing tokens this month?",
            "How do I find the whitepaper for a token?",
            "What is the total supply of a token?",
            "How can I track the price history of a token?"
        ],
        "nfts": [
            "How do I create an NFT?",
            "What are the best platforms for buying NFTs?",
            "How can I sell my NFTs?",
            "What is the value of a popular NFT collection?",
            "How do I mint an NFT on Ethereum?",
            "What are gas fees for NFTs?",
            "Can I trade NFTs on multiple platforms?"
        ],
        "smart_contracts": [
            "What is a smart contract in blockchain?",
            "How do I deploy a smart contract on Ethereum?",
            "What are the uses of smart contracts in DeFi?",
            "How do smart contracts work?",
            "What programming languages are used for smart contracts?",
            "How can I audit a smart contract?",
            "What are the benefits of using smart contracts?"
        ]
    }

    data = []

    def generate_and_append(query, label):
        examples = generate_examples_with_gpt(query)
        for example in examples:
            data.append((example, label))

    with ThreadPoolExecutor() as executor:
        futures = []
        for intent, queries in intents.items():
            for query in queries:
                futures.append(executor.submit(generate_and_append, query, intent))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating synthetic data"):
            future.result()

    df = pd.DataFrame(data, columns=["query", "label"]).sample(frac=1).reset_index(drop=True)
    return df

def save_dataset(df, filename):
    df.to_pickle(filename)

def load_dataset(filename):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        return df
    return None

def convert_to_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['query'].tolist())
    embeddings_df = pd.DataFrame(embeddings)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])
    dataset = pd.concat([embeddings_df, pd.Series(labels, name='label')], axis=1)
    return dataset, label_encoder

def train_and_evaluate_model(train_dataset, test_dataset):
    X_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values

    X_test = test_dataset.iloc[:, :-1].values
    y_test = test_dataset.iloc[:, -1].values

    model = OneVsRestClassifier(LogisticRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def save_model(model, model_filename, label_encoder, label_encoder_filename):
    joblib.dump(model, model_filename)
    joblib.dump(label_encoder, label_encoder_filename)

def load_model(model_filename, label_encoder_filename):
    model = joblib.load(model_filename)
    label_encoder = joblib.load(label_encoder_filename)
    return model, label_encoder

def predict_query(model, query, transformer_model, label_encoder):
    query_embedding = transformer_model.encode([query])
    prediction = model.predict(query_embedding)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

if __name__ == "__main__":
    model_filename = '../trained-models/multi_class_intent_classifier.pkl'
    label_encoder_filename = '../trained-models/multi_class_label_encoder.pkl'
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    dataset_filename = '../data/intent_dataset.pkl'
    
    df = load_dataset(dataset_filename)
    if df is None:
        df = generate_synthetic_data()
        save_dataset(df, dataset_filename)
        print("Generated and saved new dataset.")
    else:
        print("Loaded existing dataset.")

    if os.path.exists(model_filename) and os.path.exists(label_encoder_filename):
        model, label_encoder = load_model(model_filename, label_encoder_filename)
        print("Loaded existing model and label encoder.")
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        train_dataset, label_encoder = convert_to_embeddings(train_df)
        test_dataset, _ = convert_to_embeddings(test_df)
        model = train_and_evaluate_model(train_dataset, test_dataset)
        save_model(model, model_filename, label_encoder, label_encoder_filename)
        print("Trained and saved new model and label encoder.")

    # Example queries
    queries = [
        "How do I send Bitcoin to another wallet?",
        "What are the best yield farming strategies?",
        "Where can I find information about the latest tokens?",
        "How do I create an NFT?",
        "What is a smart contract in blockchain?"
    ]
    
    for query in queries:
        predicted_label = predict_query(model, query, transformer_model, label_encoder)
        print(f"Query: {query}\nPredicted Intent: {predicted_label}\n")
