import csv
import jieba
import bm25_code
import numpy as np
import tensorboard
import tensorflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fix_comments(file_name):
    lines = [line for line in open(file_name, 'r', encoding='utf-8').readlines()]

    fixed = open('text.txt', 'w', encoding='utf-8')

    for i ,line in enumerate(lines):
        if i == 0:
            fixed.write(line)
            prev_line = ""
            continue
        terms = line.split('\t')

        if terms[0] == prev_line.split('\t')[0]:
            if len(prev_line.split('\t')) == 6:
                fixed.write(prev_line + '\n')
                prev_line = line.strip()
            else:
                prev_line = ""
        else:
            if len(terms) == 6:
                fixed.write(prev_line + '\n')
                prev_line = line.strip()
            else:
                prev_line += line.strip()
    
    if prev_line:
        fixed.write(prev_line + '\n')
    fixed.close()

def load_data(file_name):
    book_comments = {}

    #读取数据
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item.get('book', '')
            comments = item.get('body', '')

            if not book or not comments: continue

            comments_word = jieba.lcut(comments)

            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comments_word)
        
        return book_comments

if __name__ == "__main__":
    # fix_comments('doubanbook_top250_comments.txt')
    book_comments = load_data('text.txt')
    print(book_comments)

    #生成图书 评论列表
    book_list = []
    book_comm = []
    for name, comm in book_comments.items():
        book_list.append(name)
        book_comm.append(comm)

    stop_words = [line for line in open('stopwords.txt', 'r', encoding='utf-8')]
    
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([" ".join(comments) for comments in book_comm])

    bm25_matrix = bm25_code.bm25(book_comments)
    
    tf_idf_similarity_matrix = cosine_similarity(tfidf_matrix)
    bm25_similarity_matrix = cosine_similarity(bm25_matrix)
    print(bm25_similarity_matrix.shape)
    print(tf_idf_similarity_matrix.shape)

    print(book_list)
    book_name = input("请输入图书名：")
    book_idx = book_list.index(book_name)

    tf_idf_sim_book_idx = np.argsort(-tf_idf_similarity_matrix[book_idx])[1:11]
    bm25_sim_book_idx = np.argsort(-bm25_similarity_matrix[book_idx])[1:11]


    for idx in tf_idf_sim_book_idx:
        print(f"TF_IDF《{book_list[idx]}》\t 相似度: {tf_idf_similarity_matrix[book_idx][idx]}")

    for idx in bm25_sim_book_idx:
        print(f"bm25《{book_list[idx]}》\t 相似度: {bm25_similarity_matrix[book_idx][idx]}")
