import jieba
import numpy as np
import fasttext
from gensim.models import KeyedVectors
# from tensorboard.plugins import projector
from torch.utils.tensorboard import SummaryWriter
import torch
import os 
import tensorflow as tf

#2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）

# 确保生成分词文件
def prepare_corpus():
    with open('text.txt', 'r', encoding='utf-8') as f:
        lines = f.read()
    
    with open('word_space.txt', 'w', encoding='utf-8') as f:
        f.write(" ".join(jieba.cut(lines)))

def train_fasttext(corpus_path):
    """训练Fasttext模型"""
    model = fasttext.train_unsupervised(
        corpus_path, 
        model='skipgram', 
        dim=50,       # 减小维度
        ws=4, 
        minCount=2,    # 过滤低频词
        epoch=30       # 减少迭代次数
    )
    model.save_model('fasttext_model.bin')
    return model

def calculate_similarities(model):
    """计算相似度"""
    words = model.get_words()
    words_vector = np.array([model.get_word_vector(word) for word in words])

    kv = KeyedVectors(vector_size=50)  # 与dim一致
    kv.add_vectors(words, words_vector)

    test_pairs = [
        ("自然语言", "深度"),
        ("Python", "Java"),
        ("机器", "算法")
    ]

    for w1, w2 in test_pairs:
        if w1 not in kv or w2 not in kv:
            print(f"词汇不存在：{w1}或{w2}")
            continue
        sim = kv.similarity(w1, w2)
        print(f"相似度: {w1} vs {w2} = {sim:.4f}")
        
    print("\n最邻近查询")
    for word in ["人工智能", "数据", "模型"]:
        if word not in kv:
            print(f"词汇不存在：{word}")
            continue
        print(f"\n与{word}最相近的词：")
        for sim_word, sim in kv.most_similar(word, topn=3):
            print(f"{sim_word}:{sim:.4f}")

def visualize_vectors(model, log_dir="fasttext_visualization"):
    """使用PyTorch的TensorBoard可视化"""
    # 标准化路径并强制创建目录
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 检查目录是否真正创建
    if not os.path.isdir(log_dir):
        raise RuntimeError(f"无法创建目录: {log_dir}")

    # 获取词向量
    words = model.get_words()
    word_vectors = np.array([model.get_word_vector(word) for word in words])
    vectors_tensor = torch.FloatTensor(word_vectors)

    # 创建SummaryWriter（关键修改：使用绝对路径）
    writer = SummaryWriter(log_dir=log_dir)
    
    # 添加词向量可视化
    writer.add_embedding(
        mat=vectors_tensor,
        metadata=words,
        tag="word_embeddings"
    )
    writer.close()
    print(f"可视化数据已保存到: {log_dir}")

    
if __name__ == "__main__":
    # 准备语料(取消注释如果需要)
    # prepare_corpus()
    
    model = train_fasttext("word_space.txt")
    calculate_similarities(model)
    visualize_vectors(model)
