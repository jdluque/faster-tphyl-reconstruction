#ifndef PAIR_HASH_H
#define PAIR_HASH_H

#include <utility>    // For std::pair
#include <functional> // For std::hash
#include <cstddef>    // For std::size_t

// Custom hash function for std::pair<int, int>
struct pair_hash {
    std::size_t operator () (const std::pair<int, int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);

        // A common way to combine hash values.
        // Can be adjusted based on desired distribution and collision avoidance.
        // Using a prime number multiplier and XOR is a typical approach.
        // You can use boost::hash_combine logic here if available,
        // but for a pure standard C++ approach, a combination of built-in hashes is common.
        // A simple shift and XOR:
        return h1 ^ (h2 << 1);
    }
};

#endif // PAIR_HASH_H
