import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import string
from PIL import Image

# 📚 Basic Libraries
import warnings
import scipy.stats as stats

st.set_page_config(layout = 'wide')

# ⚙️ Settings
pd.set_option('display.max_columns', None) # display all columns
warnings.filterwarnings('ignore') # ignore warnings


#Load previously trained models
rf_classifier = joblib.load('rf_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')] 

    
    tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]

    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


df = pd.read_csv('libros_limpios.csv')
balanced_data = pd.read_csv('model.csv')



image_path = "C:/Users/astri/Ironhack/Final_proyect/barra.jpeg"  
image = Image.open(image_path)
st.sidebar.image(image, use_container_width = True)



from PIL import Image

menu = st.sidebar.selectbox(
    "📚 Select a Section:",
    ("📜 Introduction", "📚 Basic Information", "📊 Interactive Visualizations", 
     "🔍 Dynamic Filters", "📖 Book Details", "🔎 Advanced Search", "🤖 Machine Learning")
)

if menu == "📜 Introduction":
    image_path_2 = "C:/Users/astri/Ironhack/Final_proyect/bc_2.png"  
    image_2 = Image.open(image_path_2)
    
    new_width = 800  
    aspect_ratio = image_2.width / image_2.height
    new_height = int(new_width / aspect_ratio * 0.54) 
    
    resized_image = image_2.resize((new_width, new_height))
    
    
    st.image(resized_image, use_container_width = True)

elif menu == "📚 Basic Information":
    
    st.title("ChapterCraft: Insights Beyond Pages 📖")
    st.write("The DB is from GoodReads, this books are the top rated by the Goodreads community. ")
    st.write("We have 9637 rows and 12 columns") 
    
    
    st.dataframe(df.head(5))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧮 Describe Table")
        st.write(df.describe().round())
    with col2:
        st.subheader("Highlights")
        st.write("The mean pages on a book are 381")
        st.write("The first book is from 800 B.C")
        st.write("At least 75% of the books were published in 2014.")
        st.write("The mean stars for the books is 4 stars")



color = '#F08080'


def top_authors_chart():
    st.subheader("✒️ Top 10 authors by books published")
    top_authors = df['author'].value_counts().head(10).sort_values(ascending=False)
    top_authors = top_authors.reset_index()
    top_authors.columns = ['author', 'count']
    fig = px.bar(top_authors, x='author', y='count', color_discrete_sequence=[color])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def stars_distribution_chart():
    st.subheader("🌟 Stars Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['stars'], kde=True, ax=ax, bins=10, color=color)
    st.pyplot(fig)

def genre_distribution_chart():
    st.subheader("📖 Popularity by Main Genre")
    genre_counts = df['genre1'].value_counts().sort_values(ascending=False)
    genre_counts = genre_counts.reset_index()
    genre_counts.columns = ['genre1', 'count']
    fig = px.bar(genre_counts, x='genre1', y='count', color_discrete_sequence=[color])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def top_books_chart():
    st.subheader("📚 Top 10 Books with the Most Pages and Highest Ratings")
    top_pages_rated = df.sort_values(['pages', 'stars'], ascending=[False, False]).head(10)
    top_pages_filtered = top_pages_rated[['title', 'author', 'pages', 'stars']]
    top_pages_filtered = top_pages_filtered.reset_index(drop=True)
    top_pages_filtered.index = top_pages_filtered.index + 1
    st.write(top_pages_filtered)

def pages_vs_ratings_chart():
    st.subheader("📊 Pages vs. Ratings Relationship")
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.scatterplot(x='pages', y='stars', data=df, ax=ax, color=color)
    st.pyplot(fig)
    st.markdown(
        """
        <div style="font-size:24px; font-weight:bold; color:white; background-color:black; padding:10px; border-radius:5px; text-align:center;">
            We can conclude that there is no relationship between the number of pages of a book and the stars they have.
        </div>
        """,
        unsafe_allow_html=True
    )



if menu == "📊 Interactive Visualizations":
    st.title("Interactive Visualizations")

    
    chart_option = st.selectbox(
        "Choose a visualization:",
        [
            "✒️ Top 10 authors by books published",
            "🌟 Stars Distribution",
            "📖 Popularity by Main Genre",
            "📚 Top 10 Books with the Most Pages and Highest Ratings",
            "📊 Pages vs. Ratings Relationship"
        ]
    )

    
    if chart_option == "✒️ Top 10 authors by books published":
        top_authors_chart()
    elif chart_option == "🌟 Stars Distribution":
        stars_distribution_chart()
    elif chart_option == "📖 Popularity by Main Genre":
        genre_distribution_chart()
    elif chart_option == "📚 Top 10 Books with the Most Pages and Highest Ratings":
        top_books_chart()
    elif chart_option == "📊 Pages vs. Ratings Relationship":
        pages_vs_ratings_chart()

    st.header('🌥️ Title Word Cloud')
    text = " ".join(df['title'].fillna("").astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


elif menu == "🔍 Dynamic Filters":
    st.title("ChapterCraft: Insights Beyond Pages 📖")
    all_genres = pd.concat([df['genre1'], df['genre2'], df['genre3']]).unique()
    genre_filter = st.selectbox("Select a Genre", sorted(all_genres))
    filtered_df = df[(df['genre1'] == genre_filter) | 
                     (df['genre2'] == genre_filter) | 
                     (df['genre3'] == genre_filter)]
    filtered_df = filtered_df.sort_values(by='stars', ascending=False)
    filtered_df_col =  filtered_df[['title', 'author', 'genre1', 'genre2','genre3','stars']]
    filtered_df_col = filtered_df_col.reset_index(drop = True)
    filtered_df_col.index = filtered_df_col.index + 1
    st.write(filtered_df_col)


elif menu == "📖 Book Details":
    st.title("ChapterCraft: Insights Beyond Pages 📖")
    st.subheader("📖 Book with the Most and Least Pages")
    max_pages = df.loc[df['pages'].idxmax()]
    min_pages = df.loc[df['pages'].idxmin()]
    st.write(f"**Book with the most pages:** {max_pages['title']} by {max_pages['author']} with ({int(max_pages['pages'])} pages)")
    st.write(f"**Book with the least pages:** {min_pages['title']} by {min_pages['author']} with ({int(min_pages['pages'])} pages)")

    
    st.subheader("⌨ Author with the Most and Least Ratings")
    max_rating_author = df.groupby('author')['ratings'].sum().idxmax()
    min_rating_author = df.groupby('author')['ratings'].sum().idxmin()
    st.write(f"**Author with the most ratings:** {max_rating_author}")
    st.write(f"**Author with the least ratings:** {min_rating_author}")

    st.subheader("🌟 Top Authors Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⭐ By Average Stars")
        top_rated_authors = df.groupby('author')['stars'].mean().sort_values(ascending=False).head(20)
        st.write(top_rated_authors)
    
    with col2:
        st.subheader("📚 By Books Published")
        top_authors = df['author'].value_counts().head(20)
        st.write(top_authors)

    with col3:
        st.subheader("💬 By Ratings Count")
        top_rated_count = df.groupby('author')['ratings'].sum().sort_values(ascending=False).head(20)
        st.write(top_rated_count)

    st.subheader("Complete information by book ")
    book = st.selectbox("Select a Book", df['title'].unique())
    book_details = df[df['title'] == book]
    st.write(book_details)


elif menu =="🔎 Advanced Search":
    st.title("ChapterCraft: Insights Beyond Pages 📖")
    best_books = df[df['stars'] > 4].sort_values(by='stars', ascending=False)
    best_books_filtered = best_books[['title', 'author', 'genre1', 'pages', 'stars']]
    best_books_filtered = best_books_filtered.reset_index(drop=True)

    best_books_filtered.index = best_books_filtered.index + 1  # Setting the index to start at 1

    st.write("**Top-rated books:**")
    st.dataframe(best_books_filtered)
   

    st.subheader("📚 Book Recommendations by Genre")
    selected_genre = st.selectbox("Select a genre:", df['genre1'].unique())
    recommended_books = df[df['genre1'] == selected_genre].sort_values(by='stars', ascending=False).head(10)
    recommended_books_filtered = recommended_books[['title', 'author', 'stars']]
    recommended_books_filtered = recommended_books_filtered.reset_index(drop=True)
    recommended_books_filtered.index = recommended_books_filtered.index + 1  
    st.write(recommended_books_filtered)

    st.subheader("📖 Book Recommendations by Author")
    selected_author = st.selectbox("Select an author:", df['author'].unique())
    author_books = df[df['author'] == selected_author].sort_values(by='stars',ascending = False)
    author_books_filtered = author_books[['title', 'stars', 'pages']]
    author_books_filtered = author_books_filtered.reset_index(drop = True)
    author_books_filtered.index = author_books_filtered.index + 1
    st.write(author_books_filtered)


elif menu == "🤖 Machine Learning":
    st.title("ChapterCraft: Insights Beyond Pages 📖")
    st.header("📖 Predict Book Genre")
    st.write("Paste a synopsis below, and we'll predict the book's genre!")

    user_input = st.text_area("Paste the book's synopsis here:")
    if user_input:
       
        synopsis = clean_text(user_input)
        
        
        new_synopsis_vectorized = tfidf_vectorizer.transform([synopsis])
        
        
        predicted_genre = rf_classifier.predict(new_synopsis_vectorized)
        
      
        st.markdown(
            f"""
            <div style="font-size:24px; font-weight:bold; color:white; background-color:black; padding:10px; border-radius:5px; text-align:center;">
                Predicted Genre: {predicted_genre[0]}
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.write("Thank you for exploring the analysis!")
