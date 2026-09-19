namespace P { int f3(std::map<std::string, int>& m, const std::string& k, std::shared_ptr<Node> p) { std::shared_ptr<Node> q = std::move(p); m[k] *= 2; return m[k + "z"] + (int)q->n.size(); } }
