---
title: tgn::TGStore::IndexRange
summary: A contiguous slice of the graph (e.g., training split).
---

# tgn::TGStore::IndexRange

A contiguous slice of the graph (e.g., training split).

`#include <tgn.h>`

## Public Functions

|      | Name                                                                                                              |
| ---- | ----------------------------------------------------------------------------------------------------------------- |
|      | **[IndexRange](Classes/structtgn_1_1TGStore_1_1IndexRange.md#function-indexrange)**() =default                    |
|      | **[IndexRange](Classes/structtgn_1_1TGStore_1_1IndexRange.md#function-indexrange)**(std::size_t s, std::size_t e) |
| auto | **[start](Classes/structtgn_1_1TGStore_1_1IndexRange.md#function-start)**() const                                 |
| auto | **[end](Classes/structtgn_1_1TGStore_1_1IndexRange.md#function-end)**() const                                     |
| auto | **[size](Classes/structtgn_1_1TGStore_1_1IndexRange.md#function-size)**() const                                   |

## Public Attributes

|             | Name                                                                         |
| ----------- | ---------------------------------------------------------------------------- |
| std::size_t | **[start\_](Classes/structtgn_1_1TGStore_1_1IndexRange.md#variable-start-)** |
| std::size_t | **[end\_](Classes/structtgn_1_1TGStore_1_1IndexRange.md#variable-end-)**     |

## Public Functions Documentation

### function IndexRange

```cpp
IndexRange() =default
```

### function IndexRange

```cpp
inline IndexRange(
    std::size_t s,
    std::size_t e
)
```

### function start

```cpp
inline auto start() const
```

### function end

```cpp
inline auto end() const
```

### function size

```cpp
inline auto size() const
```

## Public Attributes Documentation

### variable start\_

```cpp
std::size_t start_ {0};
```

### variable end\_

```cpp
std::size_t end_ {0};
```

______________________________________________________________________

Updated on 2026-03-17 at 20:21:51 -0400
