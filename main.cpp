#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <sstream>
#include <vector>
#include <cmath>
#include <float.h>
#include <limits.h>

#include <iterator>

using namespace std;
#define g(x) (1.0/(1.0+exp(-x)))
#define gprime(x) (g(x)*(1-g(x))) 

// Print out progress bar
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage, string info) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("%s \r%3d%% [%.*s%*s] ", info.c_str(), val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

// Process words to match words in the vocabulary file
string ProcessWord(string s)
{
    string t;
    // Remove punctuation.
    for (int i = 0; i < s.size(); i++)
        if (!ispunct(s[i]))
            t += s[i];

    // Convert to lower case.
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    
    return t;

}

// The feature we use is the number of occurances of a word in the file
vector<double> GetFeature(string filename, unordered_map<string, int> &word_to_idx)
{
    /*
    Args:
    - filename: path name to the data file
    - word_to_idx: a mapping where a word maps to a preceptron weight's index.
    */

    // Open file.
    ifstream in;
    in.open(filename.c_str());
    if (!in.is_open())
    {
        cerr << "File not found: " << filename << endl;
        exit(1);
    }
    
    // TODO: obtain feature
    // ---- START ENTERING CODE ---- //
    // ENTER CODE HERE //
    string line;
    getline(in, line);
    stringstream ss(line);

    string word;
    vector<string> rawWords;
    while(getline(ss, word, ' ')) {
        rawWords.push_back(word);
    }

    for (unsigned int i = 0; i < rawWords.size(); i++) {
        string tmp = ProcessWord(rawWords[i]);
        rawWords[i] = tmp;
    }

    vector<double> feature(word_to_idx.size(), 0.0);
    // cout << "size: " << rawWords.size() << endl;
    for (auto word : rawWords) {
        // cout << word << endl;
        // if it is a part of the vocab dictionary
        if (word_to_idx.find(word) != word_to_idx.end()) {
            int ind = word_to_idx[word];
            feature[ind] += 1.0;
            // cout << "ind : " << ind << endl;
            // cout << word << " : " << feature[ind] << endl;
        }
    }
    // ---- STOP ENTERING CODE ---- //
    
    return feature;
}

void save_trained_weights(vector<double> &weight)
{
    ofstream output_file("trained_weights");
    ostream_iterator<double> output_iterator(output_file, "\n");
    copy(weight.begin(), weight.end(), output_iterator);
    output_file.close();
}

int main()
{       
    // Open vocabulary file
    ifstream dict_file;
    dict_file.open("imdb.vocab");
    if (!dict_file.is_open())
    {
        cerr << "Dictionary not found." << endl;
        exit(1);
    }

    // TODO: Create word to weight index map dictionary based on the imdb.vocab
    // ---- START ENTERING CODE ---- //
    // ENTER CODE HERE //
    string line;
    unordered_map<string, int> word_to_idx;
    int i = 0;
    int N = 0;
    while (getline(dict_file, line)) {
        word_to_idx.insert(make_pair(line, i));
        i++;
        N++;
    }
    // cout << word_to_idx.size() << endl;
    // ---- STOP ENTERING CODE ---- //
    // cout << weightVec.size() << endl;
    dict_file.close();

    // initialize accuracy output file
    ofstream output_acc_file;
    output_acc_file.open("accuracy.txt");

    // TODO: Initialize perceptron weights and training parameters. 
    // Hint: check parameters in the AND perceptron code example
    // ---- START ENTERING CODE ---- //
    // ENTER CODE HERE //
    // random generate weights for each word
    vector<double> weights;
    for (int i = 0; i < N; i++) {
        // double x = (double)rand() / RAND_MAX;
        // weights.push_back(x * (1.0 - 0.0));
        weights.push_back(0.01);
    }
    vector<double> classResult;
    unordered_map<string, vector<double>> features;

    // ---- STOP ENTERING CODE ---- //

    float alpha = 1; 
    int total_epoch = 20;

    // Train the weights
    for (int epoch = 0; epoch < total_epoch; ++epoch) 
    {   
        // Read in train_list
        string file_name;
        string delimiter = "\t";
        ifstream train_file;
        train_file.open("training_list");
        float line_count = 0.0;
        cout << "Epoch " << epoch;

        // TODO: Update perceptron weights based on each data instance
        while (getline(train_file, file_name))
        {   
            // Hints (not required to follow): 
            // (1) retrieve the data and class of the data 
            // (2) obtain feature 
            // (3) update weights

            // ---- START ENTERING CODE ---- //
            // ENTER CODE HERE //
            // retrieving data and class
            stringstream ss(file_name);
            string file_dir;
            double c;
            ss >> file_dir >> c;
            classResult.push_back(c);
            // obtain feature
            vector<double> curFeature = GetFeature(file_dir, word_to_idx);
            // features.insert(make_pair(file_dir, curFeature));
            // update weights
            // cout << curFeature.size() << endl;
            double weightedSum = 0.0;
            for (unsigned int i = 0; i < curFeature.size(); i++) {
                weightedSum += weights[i] * curFeature[i];
            }

            for (unsigned int i = 0; i < curFeature.size(); i++) {
                weights[i] -= (double)alpha * (g(weightedSum) - c) * gprime(weightedSum) * curFeature[i];
            }
            // ---- STOP ENTERING CODE ---- //

            // Output current progress on training with progress bar
            line_count += 1;
            printProgress(line_count/10000, "Epoch "+ to_string(epoch));
        } 
        train_file.close();
        cout << endl;

        // Save trained weights
        save_trained_weights(weights);
        
        // Evaluate trained weights with test data
        ifstream test_file;
        test_file.open("test_list");
        float correct_pred = 0.0; // should update during evaluation
        line_count = 0.0;

        // TODO: Predict negative or positive review with current trained preceptron weights
        while (getline(test_file, file_name))
        {

            // Hint: similar to training, but without updating the weights. Should also compute prediction results.
            // ---- START ENTERING CODE ---- //
            // ENTER CODE HERE //
            stringstream ss(file_name);
            string file_dir;
            double c;
            ss >> file_dir >> c;
            // vector<double> curFeature = features[file_dir];
            vector<double> curFeature = GetFeature(file_dir, word_to_idx);
            double weightedSum = 0.0;
            for (unsigned int i = 0; i < curFeature.size(); i++) {
                weightedSum += weights[i] * curFeature[i];
            }
            // ---- STOP ENTERING CODE ---- //


            // TODO: Check if the prediction is correct. If so, increase correct_pred by one.
            // ---- START ENTERING CODE ---- //
            // ENTER CODE HERE //
            if ((g(weightedSum) > 0.5 && c == 1) || (g(weightedSum) <= 0.5 && c == 0)) {
                correct_pred += 1.0;
            }

            // ---- STOP ENTERING CODE ---- //

            // Output current progress on evaluation with progress bar
            line_count += 1.0;
            printProgress(line_count/1000, "Evaluating...");
        } 
        test_file.close();

        // Compute prediction accuracy
        float acc = (correct_pred/line_count)*100.0;
        printf("Accuracy: %.2f \n", acc);
        output_acc_file << "Epoch " << epoch << ": Accuracy = " << acc << "%" << endl;
    }
    output_acc_file.close();
}
