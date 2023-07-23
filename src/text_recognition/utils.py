import os
import re
import pandas as pd


def str2bool(v) -> bool:
    '''
        Convert string to boolean, basically used by the cmd parser.
    '''
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

def remove_comments(string: str) -> str:
    '''
        Removes comments from a string.
        Input params: string: str -> The string to remove comments from.
        Returns: The string without comments.
    '''
    string = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "" , string) # removes " /* COMMENT */ " from string
    string = re.sub(re.compile("//.*?\n" ) , "" , string) # removes " // COMMENT " from string
    string = re.sub(re.compile("#.*?\n" ) , "" , string) # removes " # COMMENT " from string    
    return string

def tokenize_space_in_label(string: str, space_token: str = "<SPACE>") -> str:
    '''
        Tokenizes the space in the label.
        Input params:
            string: str -> The string to tokenize.
            space_token: str -> The token to use for space.
        Returns:
            The string with space tokenized.
    '''
    string = string.strip()
    label = string.split(' ')[8: ] # as per annotation format
    label = space_token.join(label)
    return ' '.join(string.split(' ')[: 8]) + ' ' + label

def process_iam_handwriting_txt_files(file_path: str, 
                                      should_remove_comments: bool = True, 
                                      should_tokenize_space_in_label: bool = True,
                                      suffix: str = "_processed_file") -> str:
    '''
        Processes the IAM Handwriting dataset txt files.
        Input params:
            file_path: str -> The name of the file to process.
            should_remove_comments: bool -> Whether to remove comments or not.
            suffix: str -> The suffix to add to the processed file.
        Returns:
            The name of the processed file.
    '''
    pwd, file_name = '/'.join(file_path.split('/')[:-1]), file_path.split('/')[-1].split('.')[0]
    if f"{file_name}{suffix}.txt" not in os.listdir(pwd):
        print(f"Processing {file_path}...")
        processed_lines = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                processed_line = line
                if should_remove_comments:
                    processed_line = remove_comments(string=processed_line)
                    if processed_line == '':
                        continue
                if should_tokenize_space_in_label:
                    processed_line = tokenize_space_in_label(string=processed_line)
                processed_lines.append(processed_line + '\n')
        with open(f"{'.'.join(file_path.split('.')[:-1])}{suffix}.txt", 'w') as file:
            file.writelines(processed_lines)
    else:
        print(f"Skipped {file_path} processing as it already exists...")
    return f"{'.'.join(file_path.split('.')[:-1])}{suffix}.txt"

def get_df(txt_file_name: str, columns: list) -> pd.DataFrame:
    '''
        Reads a text file and returns a pandas DataFrame.
        Input params:
            txt_file_name: str -> The name of the txt file to read.
            columns: list -> The columns of the DataFrame.
        Returns:
            A pandas DataFrame.
    '''
    df = pd.read_csv(txt_file_name, sep=' ', header=None, names=columns)
    df.dropna(inplace=True)
    return df

def get_image_path(id: str, image_dir: str) -> str:
    '''
        Returns the image path for the given id.
        Input params:
            id: str -> The id of the image.
            image_dir: str -> The directory where the images are stored.
        Returns:
            The image path.
    '''
    id_split = id.split('-')
    p1 = id_split[0]
    p2 = p1 + '-' + id_split[1]
    image = id + '.png'
    return os.path.join(image_dir, p1, p2, image)

def process_label(label: str, space_token: str = "<SPACE>") -> str:
    '''
        Processes the label to remove the space token and '|' character.
        Input params:
            label: str -> The label to process.
            space_token: str -> The space token to remove.
        Returns:
            The processed label.
    '''
    label = label.replace(space_token, ' ')
    label = label.replace('|', ' ')
    return label

def split_data(df: pd.DataFrame, split_ratio: list = [0.6, 0.2], random_state: int = 42) -> tuple:
    '''
        Split data into train, validation and test sets.
        Input params:
            df: pandas dataframe.
            split_ratio: list of split ratios. Total items in this list can be max 2, one corresponding
                        to train split and the other to validation split. Test split would
                        be the remaining. Sum of items in `split_ratio` must be less than 1.0.
            random_state: int -> Random state to use for shuffling the dataframe.
        Returns: tuple of dataframes - (train, val) if length of `split_ratio` is 1 and (train, val, test)
                if the length of `split_ratio` is 2.
    '''
    assert len(split_ratio) <= 2, 'Length of `split_ratio` must be less than or equal to 2.'
    assert sum(split_ratio) < 1.0, 'Sum of items in `split_ratio` must be less than 1.0.'
    df = df.sample(frac=1, random_state=random_state)
    train_split_ratio = split_ratio[0]
    if len(split_ratio) == 2:
        val_split_ratio = split_ratio[1]
    else:
        val_split_ratio = None
    train_df = df.iloc[:int(len(df)*train_split_ratio)]
    if val_split_ratio is None:
        val_df = df.iloc[int(len(df)*train_split_ratio): ]
        return (train_df, val_df)
    val_df = df.iloc[int(len(df)*train_split_ratio): int(len(df)*(train_split_ratio + val_split_ratio))]
    test_df = df.iloc[int(len(df)*(train_split_ratio + val_split_ratio)): ]
    return (train_df, val_df, test_df)