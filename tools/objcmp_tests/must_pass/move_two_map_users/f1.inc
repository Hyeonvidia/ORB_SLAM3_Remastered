namespace P { int f1(std::map<std::string, int>& m, const std::string& k, std::shared_ptr<Node> p) { std::shared_ptr<Node> q = p; m[k] += (int)q->v.size(); return m[k + "x"]; } }
