import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/movies.csv")
df["features"] = df["genre"].fillna("") + " " + df["description"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["features"])
similarity = cosine_similarity(matrix)
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

def recommend(title, n=5):
    matches = df[df["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        print("Movie not found.")
        return
    title = matches.iloc[0]["title"]
    idx = indices[title]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    print(f"\nBecause you liked: {title}\n")
    for i, score in scores:
        print(f"- {df.iloc[i]['title']} | similarity: {score:.2f}")

print("Movie Recommendation System")
while True:
    title = input("\nEnter a movie title (or 'quit'): ").strip()
    if title.lower() == "quit":
        break
    recommend(title)
