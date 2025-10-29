AU_LANDMARK_MAP_68 = {
    'AU1': {'left': [21], 'right': [22], 'centre': [21, 22, 27]},
    'AU2': {'left': [18], 'right': [25], 'centre': []},
    'AU4': {'left': [21], 'right': [22], 'centre': [21, 22, 27]},
    'AU5': {'left': [37, 38], 'right': [43, 44], 'centre': []},
    'AU6': {'left': [1, 41, 31], 'right': [15, 46, 35], 'centre': []},
    'AU9': {'left': [31], 'right': [35], 'centre': [28]},
    'AU10': {'left': [31], 'right': [35], 'centre': [51]},
    'AU12': {'left': [48], 'right': [54], 'centre': []},
    'AU14': {'left': [48], 'right': [54], 'centre': []},
    'AU15': {'left': [48], 'right': [54], 'centre': []},
    'AU17': {'left': [57], 'right': [8], 'centre': []},
    'AU20': {'left': [48], 'right': [54], 'centre': [51]},
    'AU25': {'left': [], 'right': [], 'centre': [61, 64]},
    'AU26': {'left': [], 'right': [], 'centre': [61, 64]}
}

# Regenerated Action Unit to Landmark Map for a 98-point model.
# This is a logical adaptation based on the known muscle movements and a standard
# 98-point facial landmark topology (e.g., WFLW standard).
# The new points, especially on the inner lips and around the eyes, allow for
# more precise heatmap localization for certain AUs.

AU_LANDMARK_MAP_98 = {
    # Brow Units
    'AU1': {'left': [37, 38], 'right': [42, 50], 'centre': [37, 38, 42, 50, 51]},
    'AU2': {'left': [34, 41], 'right': [45, 47], 'centre': []},
    'AU4': {'left': [37, 38], 'right': [42, 50], 'centre': [37, 38, 42, 50, 51]},
    
    # Eye and Cheek Units
    'AU5': {'left': [61, 62], 'right': [70, 71], 'centre': []},
    'AU6': {'left': [3, 42, 32], 'right': [15, 47, 36], 'centre': []},
    
    # Nose and Upper Lip Units
    'AU9': {'left': [55], 'right': [59], 'centre': [29]},
    'AU10': {'left': [55], 'right': [59], 'centre': [52]},
    
    # Lip Corner and Mouth Units
    'AU12': {'left': [76], 'right': [82], 'centre': []},
    'AU14': {'left': [76], 'right': [82], 'centre': []},
    'AU15': {'left': [76], 'right': [82], 'centre': []},
    'AU20': {'left': [76], 'right': [82], 'centre': [52]},
    
    # Lower Face Units (Chin and Jaw)
    'AU17': {'left': [12, 13], 'right': [19, 20], 'centre': [85]},
    
    # Lip Parting and Jaw Drop Units (Enhanced with inner lip landmarks)
    'AU25': {'left': [], 'right': [], 'centre': [90, 94]},
    'AU26': {'left': [], 'right': [], 'centre': [90, 94]}
}