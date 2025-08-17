# PyOpenChannel GVF Examples - Quick Index

## 🚀 **Ready-to-Run Examples**

### **1. Simple GVF Example** - `gvf_simple_example.py`
**✅ WORKING** | **No dependencies** | **Perfect for beginners**
```bash
python3 examples/gvf_simple_example.py
```
- M1 and M2 profiles
- Multiple geometries  
- Clean, educational output

### **2. Basic GVF Usage** - `gvf_basic_usage.py`
**✅ WORKING** | **Optional matplotlib** | **Comprehensive tutorial**
```bash
python3 examples/gvf_basic_usage.py
```
- M1, M2, S1 profiles
- Detailed analysis
- Professional visualization

### **3. Profile Classification** - `gvf_profile_classification.py`
**✅ WORKING** | **Optional matplotlib** | **Advanced analysis**
```bash
python3 examples/gvf_profile_classification.py
```
- Automatic classification
- Multi-profile comparison
- Engineering reports

### **4. Dam Backwater Analysis** - `gvf_dam_backwater_analysis.py`
**✅ WORKING** | **Optional matplotlib/numpy** | **Professional flood analysis**
```bash
python3 examples/gvf_dam_backwater_analysis.py
```
- Multiple dam scenarios
- Flood mapping
- Bridge clearance analysis

### **5. Channel Transitions** - `gvf_channel_transitions.py`
**✅ WORKING** | **No dependencies** | **Engineering design**
```bash
python3 examples/gvf_channel_transitions.py
```
- Bridge contractions
- Slope changes
- Culvert analysis

## 📋 **Example Status Summary**

| Example | Status | Dependencies | Use Case |
|---------|--------|--------------|----------|
| Simple GVF | ✅ Complete | None | Learning, Quick tests |
| Basic Usage | ✅ Complete | matplotlib (opt) | Tutorial, Visualization |
| Profile Classification | ✅ Complete | matplotlib (opt) | Automated analysis |
| Dam Backwater | ✅ Complete | matplotlib/numpy (opt) | Flood studies |
| Channel Transitions | ✅ Complete | None | Bridge/culvert design |
| Applications Demo | ✅ Complete | None | Professional applications |

## 🧪 **Unit Tests Available**

The GVF system now includes comprehensive unit tests in the `tests/` directory:

- `tests/test_gvf_solver.py` - Core GVF solver functionality
- `tests/test_gvf_profiles.py` - Profile classification system  
- `tests/test_gvf_applications.py` - Applications module

Run tests with: `uv run --with pytest pytest tests/test_gvf_*.py`

## 🎯 **Quick Start Recommendations**

1. **New to GVF?** → Start with `gvf_simple_example.py`
2. **Want detailed analysis?** → Try `gvf_basic_usage.py`
3. **Need automated classification?** → Use `gvf_profile_classification.py`
4. **Working on flood studies?** → Run `gvf_dam_backwater_analysis.py`
5. **Designing transitions?** → Check `gvf_channel_transitions.py`

## 🔧 **All Examples Work Without matplotlib**

Every example gracefully handles missing matplotlib and provides full text-based analysis. Plotting is optional enhancement only.

---

**All examples are production-ready and demonstrate professional hydraulic engineering capabilities!** 🎉
