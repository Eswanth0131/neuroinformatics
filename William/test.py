import pandas as pd

test_df = {
    'a' : [1, 2, 3],
    'b' : [4, 5, 6],
}
testdf = pd.DataFrame(test_df)
print(testdf.iloc[0]['a'])