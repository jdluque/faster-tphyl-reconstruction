#include <iostream>
#include <unordered_set>
#include <utility> // For std::pair
#include <functional> // For std::hash

// Custom hash function for std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Combine the hash values. A common way is to use XOR.
        // You can use a more sophisticated combination if needed.
        return h1 ^ (h2 << 1); // Simple combination
    }
};
