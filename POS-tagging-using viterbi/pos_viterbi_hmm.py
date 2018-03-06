"""
CAP6640 spring 2018 - Dr.Glinos
Author: Sridevi Divya Krishna Devisetty(4436572)

This program control starts at main.
main function accepts two command line arguments ,like the names of training file and test file.
  main(sys.argv[1],sys.argv[2]) is equivalent to main('pos.train','sample_3').

  **Note: 1) Place the input files in the same directory as of the program file.
        2) Pass training file name as first argument and test file name as second.
        3) If python is installed in your system, you can run the program using the following commands.
        To run program:
            python pos_viterbi_hmm.py pos.train sample_3
        4) python has not so perfect libraries for rounding the float values, so I have tried both format and round api,
        I found using format to 6 points better than using round(6), though format does not yield perfect rounding,
         it is better than round() api. Kindly forgive if the probabilities appear not so perfect after 5th digit.
         In the end all the probabilities came perfect and the words are properly tagged.

python version: python 3.5
"""

from __future__ import division
import sys
import collections
from collections import defaultdict

END = "  "

START = ""


def print_unique_tags(tags):
    print()
    print('All Tags Observed:')
    print()
    filtered_tags = list(tags)
    filtered_tags.sort()
    for index, tag in enumerate(filtered_tags):
        print(' ', index + 1, '\t', tag)


def print_initial_distribution(tags_dict, sentence_count):
    print()
    print('Initial Distribution:\n')
    print()
    sorted_items = sorted(tags_dict.items())
    ord_items = collections.OrderedDict(sorted_items)
    for k, v in ord_items.items():
        value = v / sentence_count
        print("start [ {0}\t|\t] {1:.6f}".format(k, value))


def create_word_tag_dict(tags, words):
    word_tag_dict = {}
    for i in range(len(tags)):
        word_tag_dict[words[i]] = word_tag_dict.get(words[i], {})
        word_tag_dict[words[i]][tags[i]] = word_tag_dict[words[i]].get(tags[i], 0)
        word_tag_dict[words[i]][tags[i]] += 1
    return word_tag_dict


def print_emission_probabilities(word_tag_dict, tags):
    print()
    print("Emission Probabilities:\n\n")
    print()
    emission_prob_dict = {}
    sorted_items = sorted(word_tag_dict.items())
    ord_wrds = collections.OrderedDict(sorted_items)
    for word, tag_dict in ord_wrds.items():
        emission_prob_dict[word] = tag_dict
        for tag, count in tag_dict.items():
            tag_occurences_in_corpus = tags.count(tag)
            emission_prob = count / tag_occurences_in_corpus
            emission_prob_dict[word][tag] = float(format(emission_prob, ".6f"))
            print("\t\t{0}\t\t{1}\t\t{2:.6f}".format(word.center(10), tag.center(7), emission_prob))
    return emission_prob_dict


def lemmatize(words):
    lemmatized_words = [''] * len(words)
    for index, word in enumerate(words):
        if word.endswith(("sses", "xes", "ches", "shes")):
            new_word = word[:-2]
        elif word.endswith(("ses", "zes")):
            new_word = word[:-1]
        elif word.endswith("men"):
            new_word = word[:-2] + "an"
        elif word.endswith("ies"):
            new_word = word[:-3] + "y"
        else:
            new_word = word
        lemmatized_words[index] = new_word
    return lemmatized_words


def create_tag_tag_dict(tags_list):
    tag_tag_dict = {}
    start_tags = list()
    end_tags = list()
    for tags in tags_list:
        start_tags.append(tags[0])
        length = len(tags)
        end_tags.append(tags[length - 1])
        tags.insert(0, START)
        tags.append(END)
        for i in range(len(tags) - 1):
            tag_tag_dict[tags[i]] = tag_tag_dict.get(tags[i], {})
            tag_tag_dict[tags[i]][tags[i + 1]] = tag_tag_dict[tags[i]].get(tags[i + 1], 0)
            tag_tag_dict[tags[i]][tags[i + 1]] += 1
    return tag_tag_dict


def print_transition_probabilities(tag_tag_dict):
    print()
    print("Transition Probabilities:")
    print()
    transition_prob_dict = {}
    for key, tag_value in tag_tag_dict.items():
        transition_prob_dict[key] = tag_value
        total_count = sum(tag_value.values())
        for inner_tags, value in tag_value.items():
            transition_prob_dict[key][inner_tags] = float(
                format(transition_prob_dict[key][inner_tags] / total_count, ".6f"))
    sort_dict = sorted(transition_prob_dict.items(), key=lambda x: x[0])
    ordered = collections.OrderedDict(sort_dict)
    count = 0
    for key in ordered:
        tag_dict = transition_prob_dict[key]
        count += len(tag_dict)
        print("[{0:.6f}]".format((sum(tag_dict.values()))), end=" ")
        sorted_tags = sorted(tag_dict.items(), key=lambda x: x[0])
        ord_tags = collections.OrderedDict(sorted_tags)
        for tag, val in ord_tags.items():
            print("[{0}|{1}] {2:.6f}".format(tag, key, val), end=" ")
        print()

    return transition_prob_dict, count


def process_training_data(train_content):
    sentences = list(filter(bool, train_content.split('\n\n')))

    words_with_tags = defaultdict(list)
    for index, sentence in enumerate(sentences):
        words_with_tags[index] = sentence.split('\n')
    super_list = []

    for index, word in words_with_tags.items():
        sub_list = [[], []]
        for i, v in enumerate(word):
            x = v.split(" ")
            sub_list[0].append(x[0])
            sub_list[1].append(x[1])
        super_list.insert(index, sub_list)
    tags = list()
    words = list()
    list_of_tags = []
    start_tags_counts = defaultdict(int)
    for item in super_list:
        list_of_tags.append(item[1])
        tags += item[1]
        words += item[0]
        start_tags_counts[item[1][0]] += 1
    words = [word.lower() for word in words]
    lemmatized_words = lemmatize(words)
    word_tag_dict = create_word_tag_dict(tags, lemmatized_words)
    tag_tag_dict = create_tag_tag_dict(list_of_tags)
    print_unique_tags(set(tags))
    print_initial_distribution(start_tags_counts, len(sentences))
    emission_dict = print_emission_probabilities(word_tag_dict, tags)
    transition_dict, bigram_count = print_transition_probabilities(tag_tag_dict)
    print()
    print("Corpus Features:")
    print()
    print("Total # tags:", len(set(tags)))
    print("Total # bigrams:", bigram_count)
    print("Total # lexicals:", len(emission_dict))
    print("Total # sentences", len(sentences))
    return emission_dict, transition_dict


def viterbi_algo(emission_dict, test_words_list, transition_dict):
    final_tags = {}
    print()
    test_words = test_words_list[:]
    for word in test_words:
        if word not in emission_dict:
            index = test_words.index(word)
            final_tags[word] = 'NN'
            test_words.pop(index)

    print("Test Set Tokens Found in Corpus: ")
    print()
    for word in test_words:
        print("\t\t", word, "  :", end='\t')
        word_dict = emission_dict.get(word)
        sorted_items = sorted(word_dict.items())
        ord_words = collections.OrderedDict(sorted_items)
        for tag, val in ord_words.items():
            format_value = float(format(val, '.6f'))
            print(tag, "(", format_value, ")", end='\t')
        print()
    print()
    print("Intermediate Results of Viterbi Algorithm:")
    print()
    viterbi = []
    backpointer = []
    first_viterbi = {}
    first_backpointer = {}
    first_word_tags = emission_dict[test_words[0]].keys()
    for tag in first_word_tags:
        if tag == START:
            continue
        sensor_model = emission_dict[test_words[0]][tag]
        init_distribution = transition_dict[START][tag]
        first_viterbi[tag] = init_distribution * sensor_model
        first_backpointer[tag] = None
    total = sum(first_viterbi.values())
    print("Iteration  1 : \t\t", test_words[0], ":", end=' ')
    sorted_items = sorted(first_viterbi.items())
    ord_dict = collections.OrderedDict(sorted_items)
    for tag, prob in ord_dict.items():
        value = prob / total
        first_viterbi[tag] = value
        format_value = float(format(value, '.6f'))
        print(tag, " (", format_value, ",", first_backpointer[tag], " )", end=' ')
    print()
    final_tags[test_words[0]] = max(first_viterbi, key=first_viterbi.get)
    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)
    for index, word in enumerate(test_words[1:]):
        corpus_tags = emission_dict[word]
        prev_word_tags = viterbi.pop()
        print("Iteration", index + 2, ":", "\t\t", word, ":", end=' ')
        max_dict = {}
        max_tag_dict = {}
        viterbi_dict = {}
        for tag in corpus_tags:
            best_max = 0
            best_prev_tag = None
            for prev_tag in prev_word_tags:
                prev_prob = prev_word_tags[prev_tag]
                transit_prob = transition_dict[prev_tag].get(tag, 0.0001)
                emission_prob = emission_dict[word][tag]
                current_prob = prev_prob * transit_prob * emission_prob
                if current_prob > best_max:
                    best_max = current_prob
                    best_prev_tag = prev_tag
            max_dict[tag] = best_max
            max_tag_dict[tag] = best_prev_tag
        total = sum(max_dict.values())
        sorted_items = sorted(max_dict.items())
        ord_items = collections.OrderedDict(sorted_items)
        for max_tag, max_prob in ord_items.items():
            value = max_prob / total
            viterbi_dict[max_tag] = value
            format_value = float(format(value, ".6f"))
            print(max_tag, "(", format_value, ",", max_tag_dict[max_tag], ")", end=' ')

        viterbi.append(viterbi_dict)
        print()
        final_tags[word] = max(viterbi_dict, key=viterbi_dict.get)
        # print(viterbi_dict)
        # print("bp for word:",word, max(viterbi_dict, key=viterbi_dict.get))
    return final_tags


def calculate_prior_distribution_of_tags(tags):
    tag_list = list(filter(bool, tags))
    len_of_tags = len(tag_list)
    tag_dict = defaultdict(int)
    for tag in tag_list:
        tag_dict[tag] += 1
    print(sum(tag_dict.values()))
    print("length:", len_of_tags)
    for tag, count in tag_dict.items():
        print(tag, " ", count / len_of_tags, count)


def process_test_file(test_content):
    sentences = list(filter(bool, test_content.split('\n')))
    sentence_split = []
    for sentence in sentences:
        lower = sentence.lower()
        sentence_split.append(lower.split())
    return sentence_split


def print_tagger_output(words, tags):
    for i in range(len(words)):
        print("\t\t", "{0}".format(words[i]).center(7), " ", end='')
        print("{0}".format(tags[words[i]]).center(5))


def main(train_file, test_file):
    print('University of Central Florida')
    print()
    print('CAP6640 spring 2018 - Dr.Glinos')
    print()
    print('Viterbi Algorithm HMM Tagger by Sridevi Divya Krishna Devisetty')
    print("\n")
    train_file = open(train_file + ".txt")
    test_file = open(test_file + ".txt")
    train_content = train_file.read()
    test_content = test_file.read()
    emission_dict, transition_dict = process_training_data(train_content)
    test_sentences = process_test_file(test_content)
    for test_words in test_sentences:
        final_tags = viterbi_algo(emission_dict, test_words, transition_dict)
        print()
        print("Viterbi Tagger Output: ")
        print()
        print_tagger_output(test_words, final_tags)


# main('pos.train', 'sample_3')
main(sys.argv[1], sys.argv[2])
