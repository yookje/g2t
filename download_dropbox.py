import dropbox
 
dbx = dropbox.Dropbox(
    oauth2_access_token="sl.B9Q_IxH4bT7ttG3eBQesDjfj7zIvwRgAENEgloO5igMbGYyHhyg4DRNdotQv3P5JGzEA1Q6Qtb3tr6GY3hUhKj3fMMaAUemK5_7cAyddoAh2_BXkptxSwYwAcMYUe1eE8h0L19Yg8GHCqMQzO-X5", 
    app_key="c0s8x0w3etwbcwx", 
    app_secret="662olz27gmxzbbn", 
    timeout = None,
)

paths = [
    "tsp50_train_tar.gz",
    "tsp100_train_concorde.pkl.gz",
    "tsp500_train.tar.gz",
    "tsp1000_train.tar.gz",
    "tsp50_test_concorde.txt",
    "tsp100_test_concorde.txt",
    "tsp500_test_concorde.txt",   
    "tsp1000_test_concorde.txt",
  
    ]

for path in paths:
    dbx.files_download_to_file(f"./{path}", f"/TSPDataset_FuzzyEmbedding/{path}")
