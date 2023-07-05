import streamlit as st
import boto3
import io
import os
import re
import pandas as pd
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_words
from pythainlp.util import Trie
from googletrans import Translator
from fuzzywuzzy import fuzz

s3 = boto3.client('s3')
bucket_name = 'jb-matching-storage'
file_key1 = 'product_list_keyword.csv'

response = s3.get_object(Bucket=bucket_name, Key=file_key1)
content1 = response['Body'].read().decode('utf-8')

product_list_keyword = pd.read_csv(io.StringIO(content1))

file_key2 = 'dict_update.txt'

response = s3.get_object(Bucket=bucket_name, Key=file_key2)
content2 = response['Body'].read().decode('utf-8')

data_into_list = content2.split('\n')
data_into_list = [word.rstrip('\r') for word in data_into_list]

new_words = set(data_into_list)
words = new_words.union(thai_words())
custom_dictionary_trie = Trie(words)

# Function that processes the user input
def cut_msg(txt):
    txt = normalize(txt)
    txt = re.sub(r'\(.*?\)', lambda x: ''.join(x.group(0).split()), txt) 
    txt = re.sub(r"[^\S]?(\(.*?\))[^\S]?", r" \1 ",txt) 
    space_cut = re.split("\s", txt)
    clean_list = space_cut.copy()
    adj = []
    model = []
    name = []
    name_list = []
    special = []
    
    for word_cut in space_cut:
        bracket = re.findall("[(](.*)[)]", word_cut)
        if len(bracket) != 0:
            word = re.search("[(](.*)[)]", word_cut)
            if len(word[0]) == len(word_cut):    
                adj.append(word[1])
                clean_list.remove(word[0])
            else:
                special.append(word_cut)
    for text in clean_list:
        if len(re.findall("[a-zA-Z0-9].*", text)) > 0 and len(re.findall("\D", text)) != 0 :
            if not re.search(r"[ก-๙]", text):
                model.append(text)
            else:
                name.append(text)
        else :
            name.append(text)

    for i in name:
        x = word_tokenize(i, custom_dict = custom_dictionary_trie, engine="longest")
        for y in x:
            name_list.append(y)
    name_list += special
    
    model_check = model.copy()
    for n in model_check:
        if len(re.findall('[a-z]', n)) !=0 :
            model.remove(n)
            adj.append(n)
        elif len(re.findall('\.', n)) !=0 :
            model.remove(n)
            
    return name_list, model, adj

def keyword_search(product_cut):
    keyword = 'No Keyword'
    check = product_cut.copy()
    for word in product_cut:
        if len(re.findall("\d", word)) > 0 :
            check.remove(word)
        elif 'สี' in word and word not in data_into_list:
            check.remove(word)
    for product in check:   
        if product in data_into_list:
            keyword = product
            break
    if keyword == 'No Keyword':           
        if len(check) != 0:
            keyword = check[0]
        else:
            keyword = 'No Keyword'
    return keyword

def clean_msg(msg):
    msg = re.sub(r'#','',msg)
    msg = ' '.join(msg.split())
    msg = normalize(msg)
    return msg

def name_pred(txt):
    txt = clean_msg(txt)
    if re.search(r"[ก-๙]", txt):
        cut_txt = cut_msg(txt)
    else:
        translator = Translator()
        txt = translator.translate(txt, src='en', dest='th').text
        cut_txt = cut_msg(txt)
    keyword = keyword_search(cut_txt[0])
    model = ''.join(cut_txt[1])
    
    check_df = product_list_keyword.copy()
    word_list = list(check_df[check_df['product_keyword'] == keyword]['ItemName_trans'])
    short_word_list  = list(check_df[check_df['short_keyword'] == keyword]['ItemName_trans'])
    all_word_list = list(check_df['ItemName_trans'])
    
    score_key = []
    score_short_key = []
    score_all = []
    suggestion = []
    
    if len(model) != 0 and model in product_list_keyword['ModelName'].values:
        ans = product_list_keyword.loc[product_list_keyword['ModelName'] == model, 'ItemName_trans'].iloc[0]
        code = product_list_keyword.loc[product_list_keyword['ModelName'] == model, 'ItemCode'].iloc[0]
        txt = 'product code: ' + code + '\n\nproduct name: ' + ans + '\n\nmatching score: ' + str(100)
        suggestion.append(txt)
        status = 'Match from product model name'
    elif keyword in product_list_keyword['product_keyword'].values:
        for data in word_list:
            ratio_key = fuzz.token_set_ratio(txt, data)
            score_key.append(ratio_key)
        
        sorted_word_list = [x for _,x in sorted(zip(score_key, word_list), reverse=True)]
        sorted_score_key = sorted(score_key, reverse=True)
        max_score_key = sorted_score_key[0]
        if len(sorted_word_list) >= 5:
            suggest_match = sorted_word_list[:5]
            for i in range(len(suggest_match)):
                code = product_list_keyword.loc[product_list_keyword['ItemName_trans'] == suggest_match[i], 'ItemCode'].iloc[0]
                txt = 'product code: ' + code + '\n\nproduct name: ' + suggest_match[i] + '\n\nmatching score: ' + str(sorted_score_key[i])
                suggestion.append(txt)
        else:
            suggest_match = sorted_word_list
            for i in range(len(suggest_match)):
                code = product_list_keyword.loc[product_list_keyword['ItemName_trans'] == suggest_match[i], 'ItemCode'].iloc[0]
                txt = 'product code: ' + code + '\n\nproduct name: ' + suggest_match[i] + '\n\nmatching score: ' + str(sorted_score_key[i])
                suggestion.append(txt)
        if max_score_key > 70:
            status = 'Match from product keyword'
        else:
            status = 'Not sure'
    
    elif keyword in product_list_keyword['short_keyword'].values:
        for data in short_word_list:
            ratio_short_key = fuzz.token_set_ratio(txt, data)
            score_short_key.append(ratio_short_key)
        
        sorted_short_word_list = [x for _,x in sorted(zip(score_short_key, short_word_list), reverse=True)]
        sorted_score_short_key = sorted(score_short_key, reverse=True)
        max_score_short_key = sorted_score_short_key[0]
        if len(sorted_short_word_list) >= 5:
            suggest_match = sorted_short_word_list[:5]
            for i in range(len(suggest_match)):
                code = product_list_keyword.loc[product_list_keyword['ItemName_trans'] == suggest_match[i], 'ItemCode'].iloc[0]
                txt = 'product code: ' + code + '\n\nproduct name: ' + suggest_match[i] + '\n\nmatching score: ' + str(sorted_score_short_key[i])
                suggestion.append(txt)
        else:
            suggest_match = sorted_short_word_list
            for i in range(len(suggest_match)):
                code = product_list_keyword.loc[product_list_keyword['ItemName_trans'] == suggest_match[i], 'ItemCode'].iloc[0]
                txt = 'product code: ' + code + '\n\nproduct name: ' + suggest_match[i] + '\n\nmatching score: ' + str(sorted_score_short_key[i])
                suggestion.append(txt)
        if max_score_short_key > 70:
            status = 'Match from product short keyword'
        else:
            status = 'Not sure'
    
    else:
        for data in all_word_list:
            ratio = fuzz.token_set_ratio(txt, data)
            score_all.append(ratio)
        
        sorted_all_word_list = [x for _,x in sorted(zip(score_all, all_word_list), reverse=True)]
        sorted_score_all = sorted(score_all, reverse=True)
        max_score_all = sorted_score_all[0]
        if len(sorted_all_word_list) >= 5:
            suggest_match = sorted_all_word_list[:5]
            for i in range(len(suggest_match)):
                code = product_list_keyword.loc[product_list_keyword['ItemName_trans'] == suggest_match[i], 'ItemCode'].iloc[0]
                txt = 'product code: ' + code + '\n\nproduct name: ' + suggest_match[i] + '\n\nmatching score: ' + str(sorted_score_all[i])
                suggestion.append(txt)
        else:
            suggest_match = sorted_all_word_list
            for i in range(len(suggest_match)):
                code = product_list_keyword.loc[product_list_keyword['ItemName_trans'] == suggest_match[i], 'ItemCode'].iloc[0]
                txt = 'product code: ' + code + '\n\nproduct name: ' + suggest_match[i] + '\n\nmatching score: ' + str(sorted_score_all[i])
                suggestion.append(txt)
        status = 'Not sure (Match from entire database)'
        
    return status, suggestion

# Page for single product matching
def product_matching_page():
    st.title("Product Matching")
    product_name = st.text_input("Enter the product name")

    if st.button("Match"):
        matched_product = name_pred(product_name)

        st.write('Status:', matched_product[0])
        st.write('Suggestions:')
        for suggestion in matched_product[1]:
            st.write('- ' + suggestion)

# Page for file product matching
def file_product_matching_page():
    st.title("Product Matching")
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file is not None:
        file_extension = os.path.splitext(file.name)[1]  # Get the file extension

        if file_extension == ".csv":
            df = pd.read_csv(file)  # Read CSV file
        elif file_extension == ".xlsx":
            df = pd.read_excel(file)  # Read Excel file
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return

        # Create an empty dataframe to store the expanded rows
        expanded_df = pd.DataFrame()

        # Perform product name matching for each row in the dataframe
        for _, row in df.iterrows():
            product_name = row['product_name']
            suggestions = name_pred(product_name)[1] or ['']  # Get matching suggestions or empty list

            # Create a new row for each matching suggestion
            for suggestion in suggestions:
                new_row = {
                    'product_name': product_name,
                    'top_suggestion': suggestion
                }
                expanded_df = pd.concat([expanded_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        expanded_df[['product_code', 'matching_name', 'matching_score']] = expanded_df['top_suggestion'].str.extract(r'product code: (\w+)\n\nproduct name: ([^\n]+)\n\nmatching score: (\d+)', flags=re.IGNORECASE)

        # Display the output dataframe with selected columns
        st.write("Output Dataframe:")
        new_df = expanded_df[['product_name', 'product_code', 'matching_name', 'matching_score']]
        
        new_df['Select'] = False

        # Define the checkbox column configuration
        checkbox_column = st.column_config.CheckboxColumn("Selection")

        # Configure the column_config dictionary
        column_config = {
            "Select": checkbox_column
        }

        # Display the data editor with the checkbox column configuration
        edited_data_df = st.data_editor(new_df, column_config=column_config, disabled=['product_name', 'product_code', 'matching_name', 'matching_score'])

        # Create a submit button
        if st.button("Submit"):
            # Filter dataframe based on selected rows
            selected_rows = edited_data_df[edited_data_df["Select"]]

            # Remove the "favorite" column from the selected rows
            selected_rows = selected_rows.drop("Select", axis=1)

            # Show the selected rows
            st.write('Final Dataframe:')
            st.dataframe(selected_rows)

            # Save the DataFrame as a CSV file
            output_buffer = io.BytesIO()
            selected_rows.to_csv(output_buffer, index=False, encoding='utf-8-sig')
            output_buffer.seek(0)

            # Display the download link for the CSV file
            st.download_button(
                label="Download Output CSV",
                data=output_buffer,
                file_name="output.csv",
                mime="text/csv"
            )

# Main function to run the app
def main():
    page = st.sidebar.selectbox("Go to", ("Single Product Matching", "File Product Matching"))

    if page == "Single Product Matching":
        product_matching_page()
    elif page == "File Product Matching":
        file_product_matching_page()

if __name__ == "__main__":
    main()