# Natural-Language-Processing
Text Similarity Analysis : The Smith-Waterman local alignment algorithm - To find maximal local alignments within two text fragments.   Your program must first read, tokenize, and normalize the text in the two files. 

The program must then generate and report the edit distance and backtrace tables that are used by the algorithm.  Finally, the program must use the tables to identify and report all maximal-value alignments that are found in the tables. 

Tokenization and Normalization The text in both input files must be tokenized and normalized.  Tokenization will involve simply splitting the inputs on whitespace. This will produce raw tokens for both source and target.  Each raw token must be normalized according to the following rules. 
The normalization rules used are:  
 a) First, convert all tokens to lower case. 
 b) If a token contains alphanumeric characters and the first character or characters is (or are) non-alphanumeric, then each such leading non-alphanumeric character should be split off as a separate token; 
 c) Whether or not the second rule was applied, if the token contains alphanumeric characters and the last character or characters is (or are) non-alphanumeric, then each such trailing non-alphanumeric character should be split off as a separate token, which should be added in the correct order after application of Rule 4; 
 d) After application of the preceding rules, apply whichever one of the following sub-rules applies: 
    1.  If the token ends with "'s" (apostrophe, followed by the letter s), then split the token into two tokens: the part preceding the apostrophe, and the token "'s";  
    2. If the token end with "n't" (n-apostrophe-t), then split the token into two tokens: the part preceding the "n", and the token "not";    
    3. If the token end with "'m" (apostrophe-m), then split the token into two tokens: the part preceding the apostrophe, and the token "am"; and 
    4. If none of the preceding sub-rules applies, then accept the token as it is.  
    
Algorithm Parameters Use the following parameters for the Smith-Waterman local alignment algorithm:  
  a) Gap penalty for insertions and deletions:  -1
  b) Mismatch penalty of -1
  c) and Match score of +2.
  
  
