
# Static Code Analysis Report

## Overview
- **Files Analyzed**: 4
- **Total Lines of Code**: 530
- **Total Functions**: 42
- **Total Classes**: 13

## Complexity Analysis

### train.py
- **Total Complexity**: 25
- **Functions**: 7
- **Classes**: 0
- **Average Complexity**: 3.57

### model.py
- **Total Complexity**: 46
- **Functions**: 28
- **Classes**: 12
- **Average Complexity**: 1.15

### dataset.py
- **Total Complexity**: 13
- **Functions**: 4
- **Classes**: 1
- **Average Complexity**: 2.60

### config.py
- **Total Complexity**: 5
- **Functions**: 3
- **Classes**: 0
- **Average Complexity**: 1.67

## Documentation Analysis

### train.py
- **Coverage**: 0.0%
- **Documented Functions**: 0/7
- **Documented Classes**: 0/0

### model.py
- **Coverage**: 0.0%
- **Documented Functions**: 0/28
- **Documented Classes**: 0/12

### dataset.py
- **Coverage**: 0.0%
- **Documented Functions**: 0/4
- **Documented Classes**: 0/1

### config.py
- **Coverage**: 0.0%
- **Documented Functions**: 0/3
- **Documented Classes**: 0/0

## Quality Issues

### Long Function
- Function 'run_validation' is 59 lines long (>50)
- Function 'train_model' is 86 lines long (>50)
- Function '__getitem__' is 63 lines long (>50)

### Too Many Parameters
- Function 'run_validation' has too many parameters (10 > 7)
- Function 'build_transformer' has too many parameters (9 > 7)
- Function '__init__' has too many parameters (8 > 7)

### Poor Documentation
- Poor documentation coverage (0.0% < 50%)
- Poor documentation coverage (0.0% < 50%)
- Poor documentation coverage (0.0% < 50%)
- Poor documentation coverage (0.0% < 50%)

## Recommendations
- 🔴 **Documentation**: Add documentation. Current coverage: 0.0%
- 🔴 **Testing**: Add unit tests - no test files found
- 🟡 **Build**: Add setup.py or pyproject.toml for package management
