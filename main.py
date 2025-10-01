import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
books=pd.read_csv('books.csv')
users=pd.read_csv('users.csv')
ratings=pd.read_csv('ratings.csv')
# ================= POPULARITY BASED =================
ratings_with_name = ratings.merge(books, on='ISBN')
# Convert Book-Rating to numeric
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
ratings_with_name = ratings_with_name.dropna(subset=['Book-Rating'])
# Number of ratings
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'}, inplace=True)
# Average rating
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)
# Merge and filter popular books
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
final_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']
]
# ================= COLLABORATIVE BASED =================
# Filter active users
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]
# Filter popular books
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
# Pivot table (make sure column name matches your CSV)
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
similarity_score=cosine_similarity(pt)
def recommend(book_name, top_n=5):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_score[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]
    suggestions = []
    for i in similar_items:
        book_title = pt.index[i[0]]
        book_info = books[books['Book-Title'] == book_title].drop_duplicates('Book-Title')
        if not book_info.empty:
            suggestions.append({
                "title": book_title,
                "author": book_info['Book-Author'].values[0],
                "image": book_info['Image-URL-M'].values[0]
            })
    return suggestions

st.set_page_config(page_title="📚 Book Recommender", layout="wide")

st.title("📚 Book Recommendation System")

menu = ["🔥 Popular Books", "🤝 Collaborative Recommendations"]
choice = st.sidebar.radio("Choose Recommendation Type", menu)

if choice == "🔥 Popular Books":
    st.header("Top 50 Popular Books")

    for _, row in final_df.iterrows():
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(row['Image-URL-M'], width=100)
            with cols[1]:
                st.subheader(row['Book-Title'])
                st.write(f"✍️ Author: {row['Book-Author']}")
                st.write(f"⭐ Avg Rating: {row['avg_rating']:.2f} ({row['num_ratings']} ratings)")

elif choice == "🤝 Collaborative Recommendations":
    st.header("Find Similar Books")

    book_list = pt.index.tolist()
    selected_book = st.selectbox("Select a Book", book_list)

    if st.button("Recommend"):
        recommendations = recommend(selected_book)
        st.subheader(f"Books similar to **{selected_book}**:")

        for book in recommendations:
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(book['image'], width=100)
                with cols[1]:
                    st.subheader(book['title'])
                    st.write(f"✍️ Author: {book['author']}")
