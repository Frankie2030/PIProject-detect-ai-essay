"""
File: cluster_2_bertopic_v5.py

Description: This Python file implements text clustering using the BERTopic model. 
- The script has been customized to allow users to choose various embedding models.
- It incorporates grid search functionality to find optimal silhouette scores and davies_bouldin_score.
- The clustering process is visualized for better understanding of the results.             
             
Author: TuanBC
Email: tuanbc88@hcmut.edu.vn
Date Created: 2024-09-27
Version: 1.0

Usage Instructions:
1. Install necessary libraries (BERTopic, umap-learn, HDBSCAN, etc.).
2. Change Embed input text using embedding models such as SentenceTransformer or SimCSE.
3. Perform text clustering using the BERTopic model.
4. Visualize clustering results.

Main Functions:
- Load and preprocess data from JSON files.
- reduce dim using umap/pca. 
- run_bertopic: Performs text clustering and returns the topics along with probabilities.
- grid_search_optimization: Optimizes parameters for the model using Grid Search.
- visualize_heatmap_by_year: Visualizes topics over the years. and other visulizations.


Notes:
- Ensure you have installed the required libraries before running this file.
- Parameters can be adjusted according to user needs.


Ref:
https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html

"""

import os
import json
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
#from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import OnlineCountVectorizer
import pandas as pd
import plotly.express as px
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
import umap
import pickle
from itertools import product
import logging
import plotly.graph_objs as go
from bertopic.dimensionality import BaseDimensionalityReduction
import time
import concurrent.futures
from sklearn.metrics import pairwise_distances


# ===========================================================================================
# Tạo class Logger
class MyLogger:
    def __init__(self, log_file='app.log'):
        self.log_file = log_file
        self._initialize_logger()

    def _initialize_logger(self):
        # Check if the log file already exists; if so, append to it
        if os.path.exists(self.log_file):
            file_mode = 'a'  # Append mode
        else:
            file_mode = 'w'  # Write mode (create new file)

        # Create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Set logging level to INFO

        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # Ensure the file handler logs INFO and above

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Ensure the console handler logs INFO and above

        # Set up the logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Remove existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_message(self, message):
        self.logger.info(message)

    def change_log_file(self, new_log_file):
        """Change the log file and reinitialize the logger."""
        self.log_file = new_log_file
        self._initialize_logger()  # Reinitialize logger with the new log file

# ============================================================================================
# tạo class config để lưu các cấu hình để chạy bertopic
class Bertopic_Config:
  def __init__(self, embedding_model, dim_reduction_method,  clustering_algorithm , param_grid, embed_file_midle_name_tmp):
    self.embedding_model = embedding_model
    self.dim_reduction_method = dim_reduction_method
    self.clustering_algorithm = clustering_algorithm
    self.param_grid = param_grid
    self.embed_file_midle_name_tmp = embed_file_midle_name_tmp    

# ============================================================================================
# danh sách các biến toàn cục
logger = MyLogger()

# Enable/Disable tokenizers parallelism to avoid the warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ============================================================================================
# Danh sách các hàm xử lý
# ============================================================================================
# Hàm đọc file JSON
def load_data_json(file_path):
    logger.log_message("Load data...")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    total_articles = len(data)  # Tổng số bài viết ban đầu
    logger.log_message(f"Tổng số bài viết trong file: {total_articles}")    
    return data

# Hàm trích xuất các phương pháp trong khoảng năm
def extract_methods_and_years(data, year_start, year_end):
    methods = []
    years = []
    filtered_articles_count = 0  # Biến để đếm số bài viết sau khi lọc theo year
    for article in data:
        year = int(article['year'])
        if year_start <= year <= year_end:
            filtered_articles_count += 1  # Đếm số bài viết hợp lệ sau khi lọc
            
            # try 1 (dùng methods-phương pháp để gom cụm)
            # method_info = article.get('extracted_info_chatgpt', {})
            # if isinstance(method_info, dict):
            #     mthods = method_info.get('methods', [])
            # else:
            #     mthods = []

            # try 2 (dùng keyword để gom cụm)    
            mthods = article.get('final_keywords', [])

            # try 3 (dùng tile và abstract để gom cụm)
            #title_str = str(article.get("title_normalize", ""))
            #abstract_str = str(article.get("abstract_normalize", ""))
            #mthods = [title_str + ' ' + abstract_str]

            for method in mthods:
                if isinstance(method, str):  # Kiểm tra nếu phương pháp là chuỗi
                    methods.append(method)
                    years.append(year)
                else:
                    print(f"Phương pháp không phải chuỗi: {method}")

    # In tổng số bài viết sau khi lọc theo year
    logger.log_message(f"Tổng số bài viết lọc theo năm từ {year_start} đến {year_end}: {filtered_articles_count}")
    #log_message(f"Top 10 method in methods: {methods[:10]}")  # In thử 10 phương pháp đầu tiên để kiểm tra cấu trúc
    return methods, years

# Hàm chuẩn bị dữ liệu
def preprocess_data(methods):
    # Loại bỏ từ dừng và chuẩn hóa từ
    # vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    # vectorizer.fit(methods)  # Huấn luyện vectorizer trên dữ liệu
    # processed_methods = vectorizer.transform(methods).toarray()

    # Giữ nguyên các phương pháp dưới dạng chuỗi văn bản
    processed_methods = [str(method) for method in methods]

    return processed_methods

# Hàm lưu vector embeddings vào file
def save_embeddings(embeddings, file_path='embeddings.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.log_message(f"Embeddings saved to {file_path}")

# Hàm load vector embeddings từ file, nếu file tồn tại
def load_embeddings(file_path='embeddings.pkl'):
    try:
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.log_message(f"Embeddings loaded from {file_path}")
        return embeddings
    except FileNotFoundError:
        logger.log_message(f"No embeddings file found at {file_path}. Need to compute embeddings.")
        return None

def compute_or_load_embeddings(methods, embedding_model_name, embeddings_file='embeddings.pkl'):
    # Load embeddings nếu file tồn tại
    embeddings = load_embeddings(file_path=embeddings_file)
    
    if embeddings is None:
        # Nếu không có embeddings được lưu, thì tính toán lại embeddings
        logger.log_message("Calculating embeddings...")
        if embedding_model_name == "SimCSE":
            embedding_model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
        if embedding_model_name == 'SimCSE_large':
            embedding_model = SentenceTransformer('princeton-nlp/sup-simcse-roberta-large') # Using Supervised SimCSE from Hugging Face
        elif embedding_model_name == 'Sentence-T5':
            embedding_model = SentenceTransformer('sentence-t5-base') # Using Sentence-T5 from Hugging Face
        elif embedding_model_name == 'all-MiniLM-L6-v2':
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ValueError("Invalid embedding model. Choose 'SimCSE', 'SimCSE_large' , 'Sentence-T5', 'all-MiniLM-L6-v2'.")
        
        embeddings = embedding_model.encode(methods, show_progress_bar=True)
        
        # Lưu embeddings lại sau khi tính toán xong
        save_embeddings(embeddings, file_path=embeddings_file)
    
    return embeddings        

def reduce_dimensionality(embeddings, reduce_method="pca", n_components=50, n_neighbors=15):
    
    logger.log_message("Reduce_Dimensionality starting...")
    
    # Kiểm tra embeddings
    if not isinstance(embeddings, np.ndarray) or len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings should be a 2D numpy array, got shape: {embeddings.shape}")

    if reduce_method == "pca":
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        logger.log_message(f"Explained variance ratio (PCA): {np.sum(pca.explained_variance_ratio_):.2f}")
    elif reduce_method == "umap":
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        logger.log_message(f"+ UMAP Parameters: n_components: {umap_model.n_components},  n_neighbors: {umap_model.n_neighbors}, min_dist: {umap_model.min_dist}, metric: {umap_model.metric}, random_state: {umap_model.random_state}")
        logger.log_message(f"Reduce_Dimensionality UMAP completed with {n_components} components and {n_neighbors} neighbors.")
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose either 'pca' or 'umap'.")
    
    return reduced_embeddings  , n_components, n_neighbors

def save_reduced_embeddings_to_file(embeddings, v_components, v_neighbors, base_path):
    
    file_name = f"{base_path}tmp_reduced_embeddings_{v_components}_{v_neighbors}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(embeddings, f)
    return file_name

def load_embeddings_from_file(file_name):
    with open(file_name, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def run_bertopic(embeddings, methods, params):
   
    # min_df: minimum frequency. One important parameter to keep in mind is the min_df. 
    # This is typically an integer representing how frequent a word must be before being added to our representation
    #vectorizer_model = CountVectorizer(stop_words='english', min_df=10, ngram_range=(1, 3))
    
    vectorizer_model = OnlineCountVectorizer(stop_words="english")  # chuẩn theo Lib của Bertopic 

    # Tính toán trước ma trận khoảng cách cosine và truyền nó vào HDBSCAN với metric là 'precomputed'
    # Kiểm tra xem embeddings có NaN hoặc inf không
    # if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
    #     raise ValueError("Embeddings contain NaN or inf values. Please clean your data.")
    # Chuyển đổi embeddings sang float64 để đảm bảo tính tương thích
    # embeddings = embeddings.astype(np.float64)
    # cosine_distances = pairwise_distances(embeddings, metric='cosine') 

    cluster_model = HDBSCAN(
        min_cluster_size=params['min_cluster_size']
        , min_samples=params['min_samples']
        , cluster_selection_epsilon=params['cluster_selection_epsilon']
        , metric='euclidean' 
        #, metric='precomputed'  # đi kèm với cosine_distances
        , cluster_selection_method='eom'
        #, cluster_selection_method='leaf'   # leaf có thể tạo ra nhiều cụm nhỏ hơn, chi tiết hơn so với EOM.
        , prediction_data=True
    )

    # dùng GPU
    #hdbscan_model = cuml.cluster.HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)

    # khởi tạo empty reduction model để inject vào bertopic, vì đã reduce ở ngoài r. nếu ko inject cái này vào thì bertopic nó random reduce
    empty_reduction_model = BaseDimensionalityReduction() 

    # init bertopic model
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model
        , hdbscan_model=cluster_model
        , verbose=True  # Bật thông tin chi tiết
        , calculate_probabilities=False  #, calculate_probabilities=True # bằng True chạy rất lâu
        #, umap_model=custom_umap_model
        , umap_model=empty_reduction_model  # inject vào để skip qua bước reduce vector embed
    )

    # In ra các tham số của mô hình BERTopic
    logger.log_message("BERTopic intial completed:")
    logger.log_message(f"+ Vectorizer Model: {topic_model.vectorizer_model}")
    logger.log_message(f"+ Calculate Probabilities: {topic_model.calculate_probabilities}")
    logger.log_message(f"+ HDBSCAN Model: {topic_model.hdbscan_model}")
    logger.log_message(f"+ Verbose: {topic_model.verbose}")

    # Bắt đầu theo dõi thời gian
    start_time = time.time()
    logger.log_message("Clustering starting...")

    topics, probabilities = topic_model.fit_transform(methods, embeddings)

    #topics, probabilities = topic_model.fit_transform(methods, cosine_distances) # dùng cosine distances


    end_time = time.time()
    logger.log_message(f"Clustering completed in {end_time - start_time:.2f} seconds")

    return topic_model, topics , probabilities 

# Giảm chiều và lưu embeddings vào file pickle
def reduce_dimensionality_and_save(embeddings, bertopicConfig, output_dir, max_workers=4):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_components_values = bertopicConfig.param_grid.pop('n_components')
    n_neighbors_values = bertopicConfig.param_grid.pop('n_neighbors')

    if not n_components_values or not n_neighbors_values:
        logger.log_message("n_components hoặc n_neighbors rỗng, sẽ không giảm chiều embeddings")
        output_path = os.path.join(output_dir, f"{bertopicConfig.embed_file_midle_name_tmp}_0_0.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for v_components in n_components_values:
            for v_neighbors in n_neighbors_values:
                # Tạo đường dẫn tới file pickle
                output_path = os.path.join(output_dir, f"{bertopicConfig.embed_file_midle_name_tmp}_{v_components}_{v_neighbors}.pkl")
                
                # Kiểm tra nếu file đã tồn tại thì bỏ qua
                if os.path.exists(output_path):
                    logger.log_message(f"File {output_path} đã tồn tại. Bỏ qua giảm chiều cho n_components={v_components}, n_neighbors={v_neighbors}.")
                    continue  # Bỏ qua việc giảm chiều và tiếp tục với tổ hợp khác
                
                # Nếu không tồn tại, thực hiện giảm chiều và lưu lại
                futures.append(
                    executor.submit(
                        reduce_dimensionality_and_pickle, embeddings, v_components, v_neighbors, output_path
                    )
                )

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  
                logger.log_message(f"Reduced embeddings saved to: {result}")
            except Exception as exc:
                logger.log_message(f"reduce_dimensionality_and_save(): Generated an exception: {exc}")

def reduce_dimensionality_and_pickle(embeddings, v_components, v_neighbors, output_path):
    
    reduced_embeddings = reduce_dimensionality(
        embeddings, 
        reduce_method="umap", 
        n_components=v_components, 
        n_neighbors=v_neighbors
    )
    
    with open(output_path, 'wb') as f:
        pickle.dump(reduced_embeddings, f)

    return output_path

# Hàm chạy song song BERTopic sau khi đã giảm chiều
def run_bertopic_parallel(output_dir, methods, bertopicConfig, log_file_path, max_workers=4):
    
    best_score = -1
    best_params = None
    best_model = None
    best_topics = None
    best_probabilities  = None
    
    best_db_score = 10
    best_db_params = None  # Khởi tạo với giá trị None
    best_db_model = None
    best_db_topics = None

    log_data = []
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    stt = 0
    param_combinations = list(product(*bertopicConfig.param_grid.values()))
    logger.log_message(f"param_combinations: {param_combinations}")

    # Chỉ lấy những file bắt đầu bằng "tmp_embeddings_" và kết thúc bằng ".pkl"
    reduced_files = [
        f for f in os.listdir(output_dir) if f.startswith(f'{bertopicConfig.embed_file_midle_name_tmp}_') and f.endswith('.pkl')
    ]

    logger.log_message(f"Found {len(reduced_files)} reduced embeddings to process.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for reduced_file in reduced_files:
            v_components, v_neighbors = map(int, reduced_file.replace(f"{bertopicConfig.embed_file_midle_name_tmp}_", "").replace(".pkl", "").split('_'))
            
            with open(os.path.join(output_dir, reduced_file), 'rb') as f:
                logger.log_message(f"Load reduced embedding from pickle file {f.name}")
                reduced_embeddings = pickle.load(f)
                # Kiểm tra kiểu dữ liệu
                if isinstance(reduced_embeddings, tuple):
                    logger.log_message("Reduced embeddings are stored as a tuple. Taking the first element.")
                    reduced_embeddings = reduced_embeddings[0]  # Lấy phần tử đầu tiên nếu nó là tuple
                else:
                    logger.log_message("Reduced embeddings loaded successfully.")

                # Đảm bảo reduced_embeddings là numpy array
                if not isinstance(reduced_embeddings, np.ndarray):
                    reduced_embeddings = np.array(reduced_embeddings)

                logger.log_message(f"Reduced embedding shape: {reduced_embeddings.shape}")

   
            for params in param_combinations:
                try:
                    # Kiểm tra embeddings
                    if not isinstance(reduced_embeddings, np.ndarray) or len(reduced_embeddings.shape) != 2:
                        raise ValueError(f"Embeddings should be a 2D numpy array, got shape: {reduced_embeddings.shape}")
                    
                except ValueError as e:
                    logger.log_message(f"Error processing embedding: {e}")
                    # Có thể ghi lại thêm thông tin ở đây nếu cần

                except Exception as exc:
                    logger.log_message(f"An unexpected error occurred: {exc}")
                    
                stt += 1
                param_dict = dict(zip(bertopicConfig.param_grid.keys(), params))
                param_dict['n_components'] = v_components
                param_dict['n_neighbors'] = v_neighbors

                futures.append(
                    executor.submit(
                        run_bertopic_task, reduced_embeddings, methods, param_dict, stt
                    )
                )    

        for future in concurrent.futures.as_completed(futures):
            try:
                topic_model, topics, param_dict, probabilities, silhouette_avg, db_score, stt  = future.result()
                logger.log_message(f"STT: {stt} | Silhouette Score: {silhouette_avg}")
                logger.log_message(f"STT: {stt} | Davies Bouldin Score: {db_score}")

                # Append log data
                log_entry = {
                    "stt": stt,
                    "embedding_model": bertopicConfig.embedding_model,
                    "dim_reduction_method": bertopicConfig.dim_reduction_method,
                    "clustering_algorithm": bertopicConfig.clustering_algorithm,
                    "params": param_dict,
                    "no_of_cluster": len(set(topics)),
                    "silhouette_score": float(silhouette_avg),
                    "davies_bouldin_score": float(db_score)
                }
                log_data.append(log_entry)

                with open(log_file_path, 'w') as f:
                    json.dump(log_data, f, indent=4)

                if silhouette_avg > best_score:  # chạy từ [-1.1] càng lớn càng tốt
                    best_score = silhouette_avg
                    best_params = param_dict
                    best_model = topic_model
                    best_topics = topics
                    best_probabilities = probabilities

                if db_score < best_db_score:  # càng nhỏ càng tốt
                    best_db_score = db_score
                    best_db_params = param_dict
                    best_db_model = topic_model
                    best_db_topics = topics   

            except Exception as exc:
                logger.log_message(f"run_bertopic_parallel(): Generated an exception: {exc}")

    logger.log_message(f"Best Silhouette Score: {best_score}")
    logger.log_message(f"Best Params: {best_params}")

    if best_db_params is not None:
        logger.log_message(f"Best Davies Bouldin Score: {best_db_score}")
        logger.log_message(f"Best Davies Bouldin Params: {best_db_params}")
    else:
        logger.log_message(f"No valid Davies Bouldin score found")

    if best_topics is not None:
        best_Silhouette_entry = {
                    "stt": -1,
                    "embedding_model": bertopicConfig.embedding_model,
                    "dim_reduction_method": bertopicConfig.dim_reduction_method,
                    "clustering_algorithm": bertopicConfig.clustering_algorithm,
                    "best_params": best_params,
                    "best_no_of_cluster": len(set(best_topics)),
                    "best_silhouette_score": float(best_score)
                }
        log_data.append(best_Silhouette_entry)
    else:
        logger.log_message("No valid best_topics found during grid search.")
    
    if best_db_topics is not None:
        best_DaviesBouldin_entry = {
                    "stt": -2,
                    "embedding_model": bertopicConfig.embedding_model,
                    "dim_reduction_method": bertopicConfig.dim_reduction_method,
                    "clustering_algorithm": bertopicConfig.clustering_algorithm,
                    "best_params": best_db_params,
                    "best_no_of_cluster": len(set(best_db_topics)),
                    "best_db_score": float(best_db_score)
                }
        log_data.append(best_DaviesBouldin_entry)
    else:
        logger.log_message("No valid best_db_topics found during grid search.")    

    # Write log data to JSON file after each iteration
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=4)

    return best_model, best_topics, best_probabilities

def run_bertopic_task(reduced_embeddings, methods, param_dict, stt):
    
    topic_model, topics, probabilities = run_bertopic(reduced_embeddings, methods, param_dict)

    if len(set(topics)) > 1:
        logger.log_message(f"Silhouette Score: calculating...")
        # Generate `X` and `labels` only for non-outlier topics (as they are technically not clusters)                  
        indices = [index for index, topic in enumerate(topics) if topic != -1]
        X = reduced_embeddings[np.array(indices)]
        labels = [topic for index, topic in enumerate(topics) if topic != -1]

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, labels)
        # Compute Davies-Bouldin score
        db_score = davies_bouldin_score(X, labels)

        logger.log_message(f"+ {stt}. Cluster_Org: {len(set(topics))}, Cluster_2: {len(set(labels))}, Silhouette Score: {silhouette_avg}")
        
    else:
        silhouette_avg = -1
        db_score = 10

    return topic_model, topics , param_dict, probabilities, silhouette_avg, db_score, stt

# Hàm chính chạy toàn bộ quy trình
def grid_search_optimization(embeddings, methods, bertopicConfig, log_file_path, output_dir, max_workers=4):
    logger.log_message("Starting dimensionality reduction...")
    reduce_dimensionality_and_save(embeddings, bertopicConfig, output_dir, max_workers=max_workers)

    logger.log_message("Starting BERTopic clustering...")
    return run_bertopic_parallel(output_dir, methods, bertopicConfig, log_file_path, max_workers=max_workers)

# Hàm lưu kết quả ra file JSON
def save_cluster_results(topic_model, topics, methods, output_file):
    topic_info = topic_model.get_topic_info()
    topic_methods = {topic: [] for topic in topic_info['Topic']}
    
    # Tập hợp các phương pháp theo chủ đề
    for method, topic in zip(methods, topics):
        if topic != -1:  # Chỉ lưu những phương pháp thuộc về các chủ đề hợp lệ
            topic_methods[topic].append(method)
    
    # Chuyển đổi sang định dạng JSON
    results = []
    for topic, methods in topic_methods.items():
        results.append({
            "topic": topic,
            "methods": methods,
            "topic_info": topic_info[topic_info['Topic'] == topic].to_dict('records')
        })

    # Lưu vào file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Function to save the best results to file
def save_visualize_results(best_model, best_topics, best_probabilities, years, file_path='best_results.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump({
            'best_model': best_model,
            'best_topics': best_topics,
            'best_probabilities': best_probabilities,
            'years': years
        }, f)
    logger.log_message(f"Best results saved to {file_path}")

def load_best_results(file_path='best_results.pkl'):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.log_message(f"Best results loaded from {file_path}")
    return data['best_model'], data['best_topics'], data['best_probabilities'], data['years']

def find_best_scores(log_file_path):
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)
    
    # Initialize variables to store the best scores
    best_silhouette_score = float('-inf')
    best_davies_bouldin_score = float('inf')
    best_silhouette_config = None
    best_davies_bouldin_config = None

    # Iterate over the log data
    for entry in log_data:
        if entry['silhouette_score'] > best_silhouette_score:
            best_silhouette_score = entry['silhouette_score']
            best_silhouette_config = entry
        
        if entry['davies_bouldin_score'] < best_davies_bouldin_score:
            best_davies_bouldin_score = entry['davies_bouldin_score']
            best_davies_bouldin_config = entry
    
    # Print best configurations
    logger.log_message(f"Best Silhouette Score: {best_silhouette_score}" )
    logger.log_message(f"Best Silhouette Config: {best_silhouette_config['params']}" )

    logger.log_message(f"Best Davies-Bouldin Score: {best_davies_bouldin_score}" )
    logger.log_message(f"Best Davies-Bouldin Config: {best_davies_bouldin_config['params']}" )

# ============================================================================================
# Danh sách các hàm trực quan hóa (visualization)
# ============================================================================================
# Hàm trực quan hóa kết quả
def visualize_topics(topic_model):
    # Trực quan hóa chủ đề và các phương pháp
    topic_model.visualize_topics().show()  # Hiển thị biểu đồ UMAP của các chủ đề
    topic_model.visualize_barchart().show()  # Hiển thị biểu đồ thanh tần suất các chủ đề

# Hàm trực quan hóa heatmap theo năm cho các chủ đề bằng Plotly
def visualize_heatmap_by_year(topic_model, topics, years, probabilities , top_n, heatmap_image_path):
    # Tạo DataFrame để dễ xử lý dữ liệu
    #df = pd.DataFrame({"Topic": topics, "Year": years})
    df = pd.DataFrame({'Topic': topics, 'Year': years, 'Probabilities': probabilities.max(axis=1)})

    # Lọc bỏ các hàng có topic = -1
    df = df[df['Topic'] != -1]

    # Thay thế nhãn topic bằng tên topic
    topic_names = topic_model.get_topic_info()
    topic_mapping = {row['Topic']: row['Name'] for _, row in topic_names.iterrows()}
    df['Topic'] = df['Topic'].map(topic_mapping)
    
    # Đếm số lần xuất hiện của từng chủ đề trong từng năm
    topic_year_count = df.groupby(['Topic', 'Year']).size().unstack(fill_value=0)

    # Tính số lượng phương pháp trong mỗi chủ đề
    topic_counts = df['Topic'].value_counts().sort_values(ascending=False)

    # Lấy top N chủ đề có nhiều phương pháp nhất
    top_n_topics = topic_counts.index[:top_n]

    # Lọc dữ liệu chỉ giữ lại các chủ đề trong top N
    topic_year_count_top_n = topic_year_count.loc[top_n_topics]

    # Tạo biểu đồ heatmap sử dụng Plotly
    fig = px.imshow(
        topic_year_count_top_n,
        labels=dict(x="Year", y="Topic", color="Count"),
        # x=topic_year_count.columns,
        # y=topic_year_count.index,
        x=topic_year_count_top_n.columns,
        y=topic_year_count_top_n.index,
        color_continuous_scale="YlGnBu",
        text_auto=True,  # Show count values on the heatmap
        aspect="auto",  # Auto-size the heatmap to fit the data
        title="Heatmap:  Số lượng phương pháp theo chủ đề và năm"
    )

    # Customize the layout for better scrolling, place the year axis at the top, and adjust width
    fig.update_layout(
        autosize=False,
        height=1200,  # Set a fixed height for scrolling
        width=1200,  # Increase the width to 1500 pixels
        xaxis=dict(tickangle=-45, side="top", tickvals=topic_year_count_top_n.columns),  # Force display of all years
        yaxis=dict(tickmode='linear'),
        margin=dict(l=100, r=50, b=100, t=150),  # Adjust margins with more top space for the title
        title=f'Bertopic-Heatmap: Top {top_n} Phương pháp phân bổ theo chủ đề và năm'  # Add title with filter information
    )

    # Save the figure as a PDF, file ảnh PNG
    fig.write_image(heatmap_image_path, format='pdf')
    #fig.write_image(heatmap_image_path)
    print(f"Đã lưu biểu đồ heatmap tại: {heatmap_image_path}")

    # Hiển thị biểu đồ
    fig.show()

# Hàm trực quan hóa heatmap theo năm cho các chủ đề bằng Plotly (version 2)
def visualize_heatmap_by_year2(topic_model, topics, years, probabilities, top_n=5, heatmap_image_path='heatmap.png'):

    # Filter out topic -1 and get topic frequencies
    topic_freq = topic_model.get_topic_freq().set_index('Topic').drop(-1, errors='ignore')

    # Keep only top N topics
    top_topics = topic_freq.head(top_n).index

    # Create a DataFrame for topic-year heatmap
    df = pd.DataFrame({'Topic': topics, 'Year': years, 'Probabilities': probabilities.max(axis=1)})

    # Filter to keep only top topics
    df = df[df['Topic'].isin(top_topics)]

    # Create label with topic name and probabilities
    topic_labels = {}
    for topic in top_topics:
        topic_prob = np.mean(df[df['Topic'] == topic]['Probabilities'])
        
        # Get the representative name of the topic (first word in the representation)
        topic_info = topic_model.get_topic(topic)
        topic_name = topic_info[0][0] if topic_info else "Unnamed"
        
        # Create label with Topic number, probability, and Name
        topic_label = f"Topic {topic} ({topic_name}, Prob: {topic_prob:.2f})"
        topic_labels[topic] = topic_label

    # Replace topic numbers with labels
    df['Topic_Label'] = df['Topic'].map(topic_labels)

    # Count occurrences of each topic per year
    heatmap_data = pd.crosstab(df['Year'], df['Topic_Label'])

    # Create a heatmap using Plotly
    fig = px.imshow(heatmap_data.T, aspect="auto", color_continuous_scale="Viridis")
    fig.update_layout(title="Topic Distribution by Year with Probabilities and Names", xaxis_title="Year", yaxis_title="Topic")

    # Save heatmap
    fig.write_image(heatmap_image_path)
    print(f"Heatmap saved as {heatmap_image_path}")
    fig.show()

# Hàm trực quan hóa 3D xem xét mức độ ảnh hưởng của n_components và n_neighbors với silhouette_score
def visualize_3d_silhouette(log_file_path):
    # Load the log data
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(log_data)

    # Ensure we are only looking at relevant parameters
    df_filtered = df[['params', 'silhouette_score']].copy()

    # Extract 'n_components', 'n_neighbors', 'min_cluster_size', 'min_samples', 'cluster_selection_epsilon' from 'params'
    #df_filtered['n_components'] = df_filtered['params'].apply(lambda x: x['n_components'])
    # df_filtered['n_neighbors'] = df_filtered['params'].apply(lambda x: x['n_neighbors'])
    # df_filtered['min_cluster_size'] = df_filtered['params'].apply(lambda x: x['min_cluster_size'])
    # df_filtered['min_samples'] = df_filtered['params'].apply(lambda x: x['min_samples'])
    # df_filtered['cluster_selection_epsilon'] = df_filtered['params'].apply(lambda x: x['cluster_selection_epsilon'])

    df_filtered['n_components'] = df_filtered['params'].apply(lambda x: x['n_components'] if isinstance(x, dict) and 'n_components' in x else None)
    df_filtered['n_neighbors'] = df_filtered['params'].apply(lambda x: x['n_neighbors'] if isinstance(x, dict) and 'n_neighbors' in x else None)
    df_filtered['min_cluster_size'] = df_filtered['params'].apply(lambda x: x['min_cluster_size'] if isinstance(x, dict) and 'min_cluster_size' in x else None)
    df_filtered['min_samples'] = df_filtered['params'].apply(lambda x: x['min_samples'] if isinstance(x, dict) and 'min_samples' in x else None)
    df_filtered['cluster_selection_epsilon'] = df_filtered['params'].apply(lambda x: x['cluster_selection_epsilon'] if isinstance(x, dict) and 'cluster_selection_epsilon' in x else None)

    # Create a 3D scatter plot for silhouette score based on n_components and n_neighbors
    fig = px.scatter_3d(df_filtered, 
                        x='n_components', 
                        y='n_neighbors', 
                        z='silhouette_score', 
                        color='silhouette_score',
                        title='Silhouette Score based on n_components and n_neighbors',
                        labels={
                            'n_components': 'Number of Components',
                            'n_neighbors': 'Number of Neighbors',
                            'silhouette_score': 'Silhouette Score'
                        },
                        hover_data={
                            'min_cluster_size': True,
                            'min_samples': True,
                            'cluster_selection_epsilon': True
                        })

    # Show the 3D plot
    fig.show()

# Hàm trực quan hóa giữa số cụm và silhouette_score , davies_bouldin_score
def visualize_ClusterNo_Silhouette(log_file_path):
    # Load log data from JSON file
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)

    # Extract values for plotting
    stt = [entry['stt'] for entry in log_data]
    no_of_cluster = [entry['no_of_cluster'] for entry in log_data]
    silhouette_scores = [entry['silhouette_score'] for entry in log_data]
    davies_bouldin_scores = [entry['davies_bouldin_score'] for entry in log_data]

    # Find indices of best silhouette and davies-bouldin scores
    best_silhouette_idx = silhouette_scores.index(max(silhouette_scores))
    best_davies_bouldin_idx = davies_bouldin_scores.index(min(davies_bouldin_scores))

    # Create traces for the line plot
    cluster_trace = go.Scatter(x=stt, y=no_of_cluster, mode='lines', name='No of Clusters', line=dict(color='blue'))
    silhouette_trace = go.Scatter(x=stt, y=silhouette_scores, mode='lines', name='Silhouette Score', line=dict(color='green'))
    davies_trace = go.Scatter(x=stt, y=davies_bouldin_scores, mode='lines', name='Davies Bouldin Score', line=dict(color='red'))

    # Add markers for the best scores
    best_silhouette_marker = go.Scatter(
        x=[stt[best_silhouette_idx]], 
        y=[silhouette_scores[best_silhouette_idx]],
        mode='markers+text',
        marker=dict(color='green', size=10, symbol='star'),
        name=f'Best Silhouette (stt: {stt[best_silhouette_idx]})',
        text=[f"Silhouette: {silhouette_scores[best_silhouette_idx]:.3f}"],
        textposition='top center'
    )

    best_davies_marker = go.Scatter(
        x=[stt[best_davies_bouldin_idx]], 
        y=[davies_bouldin_scores[best_davies_bouldin_idx]],
        mode='markers+text',
        marker=dict(color='red', size=10, symbol='star'),
        name=f'Best Davies Bouldin (stt: {stt[best_davies_bouldin_idx]})',
        text=[f"Davies Bouldin: {davies_bouldin_scores[best_davies_bouldin_idx]:.3f}"],
        textposition='top center'
    )

    # Create layout
    layout = go.Layout(
        title="Grid Search Results: No of Clusters, Silhouette Score, and Davies Bouldin Score",
        xaxis_title="STT",
        yaxis_title="Scores and No of Clusters",
        legend=dict(x=0.1, y=1.2, orientation="h"),
        hovermode="x"
    )

    # Create figure and add traces
    fig = go.Figure(data=[cluster_trace, silhouette_trace, davies_trace, best_silhouette_marker, best_davies_marker], layout=layout)

    # Show the plot
    fig.show()


# ============================================================================================
# Hàm chính
# ============================================================================================
def main():
    
    # Khai báo các tham số cần thiết trong file
    year_start = 2015  # Năm bắt đầu cho phân cụm
    year_end = 2024  # Năm kết thúc cho phân cụm
    
    #extract_info = 'phuongphap' # thông tin rút trích từ paper, đang rút các methods (phuongphap), tukhoa, tomtat
    extract_info = 'tukhoa' # thông tin rút trích từ paper, đang rút các methods (phuongphap), tukhoa, tomtat
    #extract_info = 'tomtat' # thông tin rút trích từ paper, đang rút  tomtat
    
    max_workers = 1 # Khởi tạo số luồng xử lý song song

    #embedding_model_name = 'SimCSE_large'  # Choose between 'SimCSE', 'SimCSE_large' , 'Sentence-T5', 'all-MiniLM-L6-v2'
    embedding_model_name = 'Sentence-T5'
    
    dim_reduction_method = 'umap'  # Choose 'pca' or 'umap'
    clustering_algorithm = "hdbscan"  # Thuật toán phân cụm (có thể là "hdbscan" hoặc "kmeans")
    
    in_base_path = "paper_explore/result_analyze/bertopic/"
    out_base_path = "paper_explore/result_analyze/bertopic/"   # đường dẫn gốc tới folder
    prefix_name = 'o2_'  # 01 - title+abstract , o2 - tukhoa , o3 - phuongphap
    
    input_json_file_path = f'{in_base_path}list_paper_json__20240917_8_finalkeywords.json' # Đường dẫn đến file JSON chứa dữ liệu ds các bài báo
    embed_file_midle_name = f'bertopic_{year_start}_{year_end}_{extract_info}__{embedding_model_name}'
    embed_file_midle_name_tmp = f'tmp_bertopic_{year_start}_{year_end}_{extract_info}__{embedding_model_name}'
    
    embed_file_path = f'{out_base_path}{embed_file_midle_name}__embeddings.pkl'  # file vector embed để tái sử dụng
    
    midle_name =  f'bertopic_{year_start}_{year_end}_{extract_info}__{embedding_model_name}_{dim_reduction_method}_{clustering_algorithm}'
    
    output_file = f"{out_base_path}{prefix_name}{midle_name}__cluster_results.json"  # Đường dẫn tới file JSON để lưu kết quả phân cụm
    grid_search_log_file_path = f'{out_base_path}{prefix_name}{midle_name}__grid_search_log.json'  # Path to save the Grid Search log
    running_log_file_path = f'{out_base_path}{prefix_name}{midle_name}__process_tracking.log'  # Path to save the Grid Search log
    modeltopicprob_output_file = f'{out_base_path}{prefix_name}{midle_name}__best_modeltopicprob_results.pkl' # data best sau gom cụm để chạy các hàm visualize
    

    # setup logfile (process tracking)
    logger.change_log_file(running_log_file_path)

    # Bắt đầu theo dõi thời gian
    t_start_time = time.time()
    # Load and preprocess data
    data = load_data_json(input_json_file_path)
    methods, years = extract_methods_and_years(data, year_start, year_end)  # extract feature (rút các method và year trong paper)
    
    methods = preprocess_data(methods)  # tiền xử lý (call thôi chứ trong ruột ko làm gì cả)

    # Generate embeddings - tạo embeded vetor cho các methods trong paper
    embeddings = compute_or_load_embeddings(methods, embedding_model_name, embed_file_path)
    
    # Define the parameter grid for HDBSCAN
    # try 1
    # param_grid = {
    #     'min_cluster_size': [5],
    #     'min_samples': [5],
    #     'cluster_selection_epsilon': [0.1],
    #     'n_components': [5],  # UMAP/PCA components
    #     'n_neighbors': [5]    # UMAP specific parameter
    # }

    # try 2
    param_grid = {
        'min_cluster_size': [5],
        'min_samples': [5],
        'cluster_selection_epsilon': [0.1],
        'n_components': [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100],  # UMAP/PCA components
        'n_neighbors': [5]    # UMAP specific parameter
    }

    # try 3
    # param_grid = {
    #     'min_cluster_size': [5, 10, 15, 20, 25, 30],
    #     'min_samples': [5, 10, 15 , 20, 25, 30],
    #     'cluster_selection_epsilon': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    #     'n_components': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],  # UMAP/PCA components
    #     'n_neighbors': [5, 10, 15, 20, 25, 30]    # UMAP specific parameter
    # }

    # try 4 ==> ko giảm chiều chạy rất chậm, bỏ qua - ko apply. 
    # param_grid = {
    #     'min_cluster_size': [20],
    #     'min_samples': [20],
    #     'cluster_selection_epsilon': [0.1],
    #     'n_components': [],  # array rỗng - ko giảm chiều vector
    #     'n_neighbors': []    # array rỗng - ko giảm chiều vector
    # }

    bertopicConfig = Bertopic_Config(embedding_model_name, dim_reduction_method, clustering_algorithm, param_grid, 
                                     embed_file_midle_name_tmp)

    # Perform grid search for best parameters
    best_model, best_topics , best_probabilities = grid_search_optimization(embeddings, methods, bertopicConfig, grid_search_log_file_path, out_base_path, max_workers=max_workers)

    # Lưu kết quả phân cụm (string topic và ds các methods) vào file JSON
    save_cluster_results(best_model, best_topics, methods, output_file)
    save_visualize_results(best_model, best_topics, best_probabilities, years, modeltopicprob_output_file)

    # In kết quả
    logger.log_message(f"Số lượng chủ đề phát hiện: {len(set(best_topics))}")

    # Kết thúc theo dõi thời gian
    t_end_time = time.time()
    t_processing_time = t_end_time - t_start_time

    # Convert minutes to hours and minutes
    t_hours = int(t_processing_time // 3600)  # Lấy số giờ
    t_minutes = int((t_processing_time % 3600) // 60)  # Lấy số phút
    t_seconds = int(t_processing_time % 60)  # Lấy số giây

    logger.log_message(f"Finished processing (total) in {t_hours} hours, {t_minutes} minutes, {t_seconds} seconds")       


if __name__ == "__main__":  
    
    main()

    #visualize_3d_silhouette('paper_explore/result_analyze/bertopic/o1_bertopic_2015_2024_phuongphap__SimCSE_large_umap_hdbscan_phuongphap__grid_search_log.json')

    #find_best_scores('paper_explore/result_analyze/bertopic/Result_20240928_cluster_2_bertopic/o1_bertopic_2015_2024_phuongphap__Sentence-T5_umap_hdbscan__grid_search_log.json')