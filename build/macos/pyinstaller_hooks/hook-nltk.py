# Disable the stock nltk hook — it pulls sklearn/pandas and can crash Analysis
# on machines with a numpy ABI mismatch. H3-WS does not use nltk.
excludedimports = ["nltk"]
