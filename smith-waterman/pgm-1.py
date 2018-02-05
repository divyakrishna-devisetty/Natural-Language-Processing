"""
CAP6640 spring 2018 - Dr.Glinos
Author: Sridevi Divya Krishna Devisetty(4436572)

All the methods of this program have python pep8 standard method descriptions explaining their functionality, input and
return values of each method.

This program control starts at main. main function accepts two command line arguments like the names of input text files
  main(sys.argv[1],sys.argv[2]) is equivalent to main('gene-src','gene-tgt').

**Note: 1) Place the input files in the same directory as of the program file.
        2) Pass source file name as first argument and target as second.
        3) If python is installed in your system, you can run the program using the following commands.
        To run gene:
            python pgm-1.py gene-src gene-tgt
        To run pepper:
            python pgm-1.py pepper-src pepper-tgt
        To run shake:
            python pgm-1.py shake-src shake-tgt

python version: python 3.5
used packages: numpy,re
"""

import sys
import re
import numpy as np
import copy

# constants
gap_penalty = -1
mismatch_penalty = -1
match_score = 2
END, DIAG, UP, LEFT = ("", "DI", "UP", "LT")


def tokenize(text):
    """
    This method splits the input text based on space and returns the string joined by a space
    :param text: a string
    :return: a string
    """
    list_of_tokens = text.split(' ')
    return ' '.join(list_of_tokens)


def split_alphanumerics(string_to_process):
    """
    This method splits the given string and process each alphanumeric token with non-alphanumeric characters,
    and splits them as separate strings and returns it.
    :param string_to_process
    :return: a string split with separated leading and trailing non alphanumerics
    """
    strings = []
    regex = r'(^\W+|\W+$)'

    list_of_tokens = string_to_process.split()
    # Here re.split() gives you both matched and unmatched tokens list. The else part just splits the alphanumeric
    # string with leading and trailing non-alphanumeric characters.
    for token in list_of_tokens:
        match = re.split(regex, token)
        if match:
            if len(match) == 1:
                strings.append(''.join(match))
            else:
                if any(c.isalnum() for c in match):
                    for sub_token in match:
                        if not sub_token.isalnum():
                            index = match.index(sub_token)
                            match[index] = ' '.join(sub_token)

                strings.append(' '.join(match))
    final_string = ' '.join(strings)
    return final_string


def apply_normalize_rules(split):
    """
    This method applies the rule num 4 in the assignment.
    :param split: a string with separated non alphanumerics
    :return: a normalized string
    """
    list_of_tokens = split.split()
    for token in list_of_tokens:
        index = list_of_tokens.index(token)
        if token.endswith("'s"):
            list_of_tokens[index] = " ".join(list(filter(bool, token.partition("'s"))))
        if token.endswith("n't"):
            replace_str = list(filter(bool, [x.replace("n't", "not") for x in token.partition("n't")]))
            list_of_tokens[index] = " ".join(replace_str)
        if token.endswith("'m"):
            replace_str = list(filter(bool, [x.replace("'m", "am") for x in token.partition("'m")]))
            list_of_tokens[index] = " ".join(replace_str)
    return " ".join(list_of_tokens)


def normalize(text):
    """
    This method carries out two operations on the given text.First it converts text to lower case and then
    separates leading and trailing non-alphanumerics of a alphanumeric word and then applies rule-4 of normalization.
    :param text: tokenized text.
    :return: normalized text
    """
    lower_cased = text.lower()
    split = split_alphanumerics(lower_cased)
    normalized = apply_normalize_rules(split)
    return normalized


def calculate_score(similar, edit_table, i, j):
    """
    This method calculates score to be filled in edit distance table.
    if there is a match it adds match_score of 2 to diagonal value else it adds mismatch_penalty of -1.
    It adds gap penalty of -1 to top and left values. To avoid negatives we take max with 0.
    :param similar: a boolean which states if two words are matching.
    :param edit_table: edit distance matrix
    :param i: source word index
    :param j: target word index
    :return: max of 0, diagonal, top and left values.
    """
    temp_map = {}
    if similar:
        diag_value = match_score + edit_table[i - 1][j - 1]
    else:
        diag_value = mismatch_penalty + edit_table[i - 1][j - 1]

    top_value = edit_table[i - 1][j] + gap_penalty

    left_value = edit_table[i][j - 1] + gap_penalty
    temp_map[DIAG] = diag_value
    temp_map[UP] = top_value
    temp_map[LEFT] = left_value
    return max(0, diag_value, top_value, left_value), temp_map


def get_trace(dir_map, score):
    """
    Based on the score this method provides the direction which yielded the score.
    :param dir_map: map of score values with each direction as key.
    :param score: the value in edit-distance table.
    :return: returns the direction.
    """
    if score == 0:
        return END
    if score == dir_map[UP]:
        return UP
    if score == dir_map[LEFT]:
        return LEFT
    if score == dir_map[DIAG]:
        return DIAG


def create_edit_dist_and_backtrace_table(source_content, target_content):
    """
    This method creates and fills values to edit-distance and backtrace tables.
    :param source_content: normalized source string
    :param target_content: normalized target string
    :return: returns edit distance and backtrace table.
    """
    rows = len(source_content) + 1
    columns = len(target_content) + 1
    edit_table = [[0 for col in range(columns)] for row in range(rows)]
    backtrace_table = [['' for col in range(columns)] for row in range(rows)]
    for i in range(1, rows):
        for j in range(1, columns):
            similar = source_content[i - 1] == target_content[j - 1]
            score, dir_map = calculate_score(similar, edit_table, i, j)
            edit_table[i][j] = score
            backtrace_table[i][j] = get_trace(dir_map, score)
    return edit_table, backtrace_table


def align_string(aligned_seq1, aligned_seq2):
    """
    This method aligns source and target strings based on insertion, deletion and substitution.
    If - in source we tag it as insertion and - in target we mark it as deletion and rest unmatched
    we tag it as substitution.
    :param aligned_seq1: source sequence
    :param aligned_seq2: target sequence
    :return: a list with strings indicating i-insertion, d-deletion and s-substitution.
    """
    alignment_string = []
    for base1, base2 in zip(aligned_seq1, aligned_seq2):
        if base1 == base2:
            alignment_string.append(' ')
        elif '-' in base1:
            alignment_string.append('i')
        elif '-' in base2:
            alignment_string.append('d')
        else:
            alignment_string.append('s')
    return alignment_string


def traceback(start_pos, seq1, seq2, backtrace):
    """
    This method traces back from the max_value in edit-distance table till it reaches end, the moves are evaluated
    using the values in backtrace table and  aligned source and target sequences are returned accordingly.
    :param start_pos: index of max_value in edit distance table
    :param seq1: source
    :param seq2: target
    :param backtrace: backtrace table
    :return: aligned source and targets.
    """
    aligned_seq1 = []
    aligned_seq2 = []
    x, y = start_pos
    move = backtrace[x][y]
    while move != END:
        if move == DIAG:
            aligned_seq1.append(seq1[x - 1])
            aligned_seq2.append(seq2[y - 1])
            x -= 1
            y -= 1
        elif move == UP:
            aligned_seq1.append(seq1[x - 1])
            aligned_seq2.append('-')
            x -= 1
        else:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[y - 1])
            y -= 1

        move = backtrace[x][y]

    return aligned_seq1[::-1], aligned_seq2[::-1], (x, y)


def print_table(table, source, target):
    """
    This method is just to format the edit distance and backtrace tables.
    :param table: edit distance or back trace table
    :param source: source sequence
    :param target: target sequence
    :return: formatted tables
    """
    matrix = copy.deepcopy(table)
    src = source[:]
    tgt = target[:]
    src.insert(0, '#')
    tgt.insert(0, '#')
    for i in range(len(src)):
        row = matrix[i]
        row.insert(0, i)
        row.insert(1, src[i])
    first_row = [' ', ' ']
    second_row = [' ', ' '] + tgt
    for i in range(len(tgt)):
        first_row.append(i)
    matrix.insert(0, first_row)
    matrix.insert(1, second_row)

    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(rows):
        for col in range(cols):
            label = str(matrix[row][col])
            if len(label) > 3:
                element = label[:3]
            else:
                element = label
            print('{0}'.format(element).center(5), end='  ')
        print()


def print_maximum_indices(indices):
    """
    This method is used to print the indices at which maximal local alignment happens.
    :param indices: [x,y] in edit distance table at which maximal value occurs
    """
    for index in zip(indices[0], indices[1]):
        print("\t[\t", index[0], ",", index[1], "\t]")


def main(source_file, target_file):
    """
    The main method first reads input files, tokenizes src and tgt sequences and normalizes them. It then
    uses normalized strings to fill edit distance and backtrace table and finds the alignment strings by traceback
    and prints them accordingly.
    :param source_file: source_file name
    :param target_file: target_file name
    """
    print('University of Central Florida')
    print("\n")
    print('CAP6640 spring 2018 - Dr.Glinos')
    print("\n")
    print('Text Similarity Analysis by Sridevi Divya Krishna Devisetty')
    print("\n")
    # if not sys.argv[1:]:
    #     print('Please provide source and target file names as arguments')
    #     sys.exit(1)

    print("Source file:", source_file + ".txt")
    print("\n")
    print("Target file:", target_file + ".txt")
    print("\n")

    source_content = open(source_file + '.txt', "r")
    target_content = open(target_file + '.txt', "r")

    print('Raw Tokens:')
    print("\n")
    source_tokens = tokenize(source_content.read())
    target_tokens = tokenize(target_content.read())
    print('Source >', source_tokens)
    print("\n")
    print('Target >', target_tokens)
    print("\n")

    print('Normalized Tokens:')
    print("\n")
    normalized_source = normalize(source_tokens)
    normalized_target = normalize(target_tokens)
    print('Source >', normalized_source)
    print("\n")
    print('Target >', normalized_target)
    print("\n")

    source_str = normalized_source.split()
    target_str = normalized_target.split()
    edit_dist, backtrace = create_edit_dist_and_backtrace_table(source_str, target_str)

    print("Edit Distance Table:")
    print_table(edit_dist, source_str, target_str)
    print("\n")
    print("Backtrace Table:")
    print_table(backtrace, source_str, target_str)
    print("\n")

    edit_dist_array = np.array(edit_dist)
    print("Maximum value in distance table:", edit_dist_array.max())
    indices = np.where(edit_dist_array == edit_dist_array.max())
    print("\n")
    print("Maxima:")
    print_maximum_indices(indices)
    print("\n")

    align_seq = 0

    for index in zip(indices[0], indices[1]):
        src_aligned, tgt_aligned, pos = traceback(index, source_str, target_str, backtrace)
        digit = len(str(max(src_aligned + tgt_aligned, key=len)))
        print("Maximal Similarity Alignments: ")
        print("\t Alignment", align_seq, "( length ", len(src_aligned), "):")
        print("\t\tSource at\t\t{0}\t:".format(pos[0]), end=' ')
        for src in src_aligned:
            print("{0}".format(src).center(digit + 3), end=' ')
        print()
        print("\t\tTarget at\t\t{0}\t:".format(pos[1]), end=' ')
        for tgt in tgt_aligned:
            print("{0}".format(tgt).center(digit + 3), end=' ')
        print()
        alignment_str = align_string(src_aligned, tgt_aligned)
        print("\t\tEdit action\t\t\t:", end=' ')
        for align in alignment_str:
            print("{0}".format(align).center(digit + 3), end=' ')
        print()
        align_seq += 1


# main('gene-src', 'gene-tgt')
# main('pepper-src', 'pepper-tgt')
main('shake-src', 'shake-tgt')
# main(sys.argv[1], sys.argv[2])
