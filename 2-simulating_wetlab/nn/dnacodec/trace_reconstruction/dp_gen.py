def msa_loop(n):
    code = "\t// Main MSA loop\n"
    for i in range(0, n):
        code += "for (i[" + str(i) + "] = 0; i[" + str(i) + "] <= l[" + str(i) + "]; ++i[" + str(i) + "]) {\n"
    code += "\nif("
    for i in range(0, n - 1):
        code += "(i[" + str(i) + "] == 0) && "
    code += "(i[" + str(n - 1) + "] == 0)) {\n"
    code += "score" + n * "[0]" + " = 0;\n"
    code += "track" + n * "[0]" + " = 0;\n} else {\n"
    code += "score" + build_index(n) + " = 255;\n"
    code += """for (int j = (1 << N) - 1; j > 0; --j) {
                                bitset<N> bin(j);
                                int count[5], maxi = 0;

                                for (int m = 0; m < 5; ++m)
                                    count[m] = 0;

                                for (int k = 0; k < N; ++k) {
                                    cost_array[k] = get_char((int) (bin[N - 1 - k]), i[k], s.at(k));
                                    int id = get_id(cost_array[k]);
                                    count[id]++;
                                    if (maxi < count[id]) {
                                        maxi = count[id];
                                    }
                                }

                                if (count[0] == N){
                                    maxi = 0;
                                }

                                bool valid = true;
                                for (int k = 0; k < N; ++k) {
                                    if (i[k] - (int)(bin[N - 1 - k]) < 0) {
                                        valid = false;
                                        break;
                                    }
                                }

                                if (!valid)
                                    continue;
                                \n"""
    code += "unsigned char sc = score"
    for i in range(0, n):
        code += "[max(0, i[" + str(i) + "] - (int) (bin[N - 1 - " + str(i) + "]))]"
    code += " + (unsigned char)(N - maxi);\n"
    code += "if (score" + build_index(n) + " > sc) {\n"
    code += "score" + build_index(n) + " = sc;\n"
    code += "track" + build_index(n) + " = (unsigned char) (j);\n}\n}\n}\n"
    code += "}\n" * n

    return code


def allocation(n):
    code = "\t// Memory Allocation\n"
    code += "unsigned char " + ('*' * n) + "score = new unsigned char " + ('*' * (n - 1)) + "[l[0] + 1];\n"
    code += "unsigned char " + ('*' * n) + "track = new unsigned char " + ('*' * (n - 1)) + "[l[0] + 1];\n\n"

    code += allocate_recursive(n, 1)
    return code


def freeing(n):
    code = "\t// Deleting score matrix\n"
    code += free_recursive(n, 1)
    code += "delete [] score;\ndelete [] track;\n\n"
    return code


def build_index(n):
    index = ""
    for i in range(0, n):
        index += "[i[" + str(i) + "]]"
    return index


def index(n):
    index = ""
    for i in range(0, n):
        index += "[l[" + str(i) + "]]"
    return index


def backtrack(n):
    code = "\t// Backtrack loop (Generating alignments)\nfor (int n = 0; n < N * L; ++n) {\n"
    code += "if ("
    for i in range(0, n - 1):
        code += "(i[" + str(i) + "] <= 0) && "
    code += "(i[" + str(n - 1) + "] <= 0))\n"
    code += "break;\n"
    code += "bitset<N> bin(track" + build_index(n) + ");\n"
    code += """for (int k = 0; k < N; ++k) {
            int d = (int) bin[N - 1 - k];
            s_aligned[k] += get_char(d, i[k], s.at(k));
            i[k] -= d;
        }
    }\n\n"""
    return code


def allocate_recursive(n, k):
    code = ""
    index = ""
    for i in range(0, k):
        index += "[i" + str(i + 1) + "]"
    code += "for (int i" + str(k) + " = 0; i" + str(k) + " < l[" + str(k - 1) + "] + 1; ++i" + str(k) + ") {\n"
    code += "score" + index + " = new unsigned char " + (n - k - 1) * '*' + "[l[" + str(k) + "] + 1];\n"
    code += "track" + index + " = new unsigned char " + (n - k - 1) * '*' + "[l[" + str(k) + "] + 1];\n"
    if k != (n - 1):
        code += allocate_recursive(n, k + 1)
    code += "}\n"
    return code


def free_recursive(n, k):
    code = ""
    index = ""
    for i in range(0, k):
        index += "[i" + str(i + 1) + "]"
    code += "for (int i" + str(k) + " = 0; i" + str(k) + " < l[" + str(k - 1) + "] + 1; ++i" + str(k) + ") {\n"
    if k != (n - 1):
        code += free_recursive(n, k + 1)
    code += "delete [] score" + index + ";\n"
    code += "delete [] track" + index + ";\n"
    code += "}\n"
    return code


def generate_msa(n):
    code = """#include <iostream>
              #include <vector>
              #include <bitset>
              #include <string>
              #include <algorithm>
              #include <time.h>
              #include <fstream>


              #define N """
    code += str(n)
    code += """\n#define P 0.1
              #define L 15
              #define T 100

              using namespace std;

              char get_char(int d, int i, string s) {
                  return (d == 0) || (d > i) ? '-' : s.at(static_cast<unsigned int>(i - d));
              }

              int get_id(char x) {
                  switch (x) {
                      case 'A':
                          return 1;
                      case 'C':
                          return 2;
                      case 'G':
                          return 3;
                      case 'T':
                          return 4;
                      default:
                          return 0;
                  }
              }

              int msa(const vector<string> &s, const int l[], char* param, char* path) {  \n\n"""
    code += allocation(n)
    code += """int i[N];
    char cost_array[N];

    for (int k = 0; k < N; ++k) {
        i[k] = 0;
        cost_array[k] = '-';
    }

    //cout << "Aligned Noisy DNAs" << endl;

    clock_t tStart = clock();\n\n"""
    code += msa_loop(n)
    code += "//cout << (int) score" + index(n) + " << endl;\n"
    code += """//cout << "Time of MSA loop: " << (double) (clock() - tStart) / CLOCKS_PER_SEC << " sec" << endl;

    vector<string> s_aligned;

    for (int k = 0; k < N; ++k) {
        i[k] = l[k];
        string str;
        s_aligned.push_back(str);
    }\n\n"""
    code += backtrack(n)
    code += freeing(n)
    code += """     ofstream out;
    string out_name = "";
    out_name.append(path);
    out_name.append("/res");
    out_name.append(param);
    out_name.append(".txt");
    out.open(out_name.c_str());


    for (int k = 0; k < N; ++k) {
        reverse(s_aligned[k].begin(), s_aligned[k].end());
        //cout << s_aligned[k] << endl;
        out << s_aligned[k] << endl;
    }

    out.close();
    return 0;
}


int main(int argc, char* argv[]) {
    vector<string> s;
    ifstream in;
    int l[N];

    string in_name = "";
    in_name.append(argv[2]);
    in_name.append("/test");
    in_name.append(argv[1]);
    in_name.append(".txt");
    in.open(in_name.c_str());
    //cout << in_name << endl;


    for (int i = 0; i < N; ++i) {
        string v;
        getline(in, v);
        //cout << v << endl;
        l[i] = v.length();
        s.push_back(v);
    }
    in.close();
    in.close();
    msa(s, l, argv[1], argv[2]);

    return 0;
}

"""
    return code
