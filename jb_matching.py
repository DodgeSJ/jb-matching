import streamlit as st
from streamlit_option_menu import option_menu
import boto3
import io
import os
import re
import clipboard
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
        st.session_state.matched_product = matched_product

    # Clear session state if clear cache button is clicked
    if st.button("Clear"):
        st.session_state.clear()

    if "matched_product" in st.session_state:
        matched_product = st.session_state.matched_product

        st.write('Status:', matched_product[0])
        st.write('Suggestions:')
        for suggestion in matched_product[1]:
            col1, col2 = st.columns([9, 1])
            col1.write('- ' + suggestion)
            with col2:
                button_id = suggestion.replace(" ", "_")
                st.button(label='Copy', key=button_id, on_click=copy_text_to_clipboard, args=(suggestion,))

def copy_text_to_clipboard(text):
    pattern = r'product code:\s*(\w+)'
    matches = re.findall(pattern, text)
    if matches:
        code = matches[0]
        clipboard.copy(code)

# Page for file product matching
def file_product_matching_page():
    st.title("Product Matching")
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file is not None:
        file_extension = os.path.splitext(file.name)[1]

        if file_extension == ".csv":
            df = pd.read_csv(file)
        elif file_extension == ".xlsx":
            df = pd.read_excel(file)
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return

        # Option 1: Auto-Select Highest Matching Score
        auto_select = st.checkbox("Auto-Select Highest Matching Score")

        if auto_select:
            df['top_suggestion'] = df['input_name'].apply(lambda input_name: name_pred(input_name)[1][0] if name_pred(input_name)[1] else '')
            df[['product_code', 'product_name', 'matching_score']] = df['top_suggestion'].str.extract(r'product code: (\w+)\n\nproduct name: ([^\n]+)\n\nmatching score: (\d+)', flags=re.IGNORECASE)

            st.write("Output Dataframe:")
            output_df = df[['input_name', 'product_code', 'product_name', 'matching_score']]
            st.dataframe(output_df)

            output_buffer = io.BytesIO()
            output_df.to_csv(output_buffer, index=False, encoding='utf-8-sig')
            output_buffer.seek(0)

            st.download_button(
                label="Download Output CSV",
                data=output_buffer,
                file_name="output.csv",
                mime="text/csv"
            )

        # Option 2: Manual Selection
        manual_select = st.checkbox("Manual Selection")

        if manual_select:
            expanded_df = pd.DataFrame()

            for _, row in df.iterrows():
                input_name = row['input_name']
                suggestions = name_pred(input_name)[1] or ['']

                for suggestion in suggestions:
                    new_row = {
                        'input_name': input_name,
                        'top_suggestion': suggestion
                    }
                    expanded_df = pd.concat([expanded_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            expanded_df[['product_code', 'product_name', 'matching_score']] = expanded_df['top_suggestion'].str.extract(r'product code: (\w+)\n\nproduct name: ([^\n]+)\n\nmatching score: (\d+)', flags=re.IGNORECASE)
            new_df = expanded_df[['input_name', 'product_code', 'product_name', 'matching_score']]

            st.write("Output Dataframe:")
            new_df = expanded_df[['input_name', 'product_code', 'product_name', 'matching_score']]
            new_df['Select'] = False

            checkbox_column = st.column_config.CheckboxColumn("Selection")
            column_config = {
                "Select": checkbox_column
            }

            edited_data_df = st.data_editor(new_df, column_config=column_config, disabled=['input_name', 'product_code', 'product_name', 'matching_score'])

            if st.button("Submit"):
                selected_rows = edited_data_df[edited_data_df["Select"]]
                output_df = selected_rows.drop("Select", axis=1)

                st.write('Final Dataframe:')
                st.dataframe(output_df)
        
                output_buffer = io.BytesIO()
                output_df.to_csv(output_buffer, index=False, encoding='utf-8-sig')
                output_buffer.seek(0)

                st.download_button(
                    label="Download Output CSV",
                    data=output_buffer,
                    file_name="output.csv",
                    mime="text/csv"
                )

selected = option_menu(
    menu_title = 'Jenbunjerd product matching',
    options = ['Single product matching', 'File product matching'],
    default_index = 0,
    icons = ['1-circle', 'file-excel'],
    menu_icon = 'card-list',
    orientation = 'horizontal'
    )

if selected == 'Single product matching':
    product_matching_page()
if selected == 'File product matching':
    file_product_matching_page()