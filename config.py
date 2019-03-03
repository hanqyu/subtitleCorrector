CONFIG = {
    'gradient': {
        'kernel_size_row': 3,
        'kernel_size_col': 3
    },

    'resize': False,

    'resize_origin': {
        'standard_width': 1400,
        'standard_height': 800,
    },

    'threshold': {
        'mode': 'mean',  # global, mean, gaussian
        'block_size': 5,  # Threshold (Odd number !!)
        'subtract_val': 15,  # Threshold
    },

    'remove_line': {
        'threshold': 200,  # Long Line Remove Precision
        'min_line_length': 57,  # Long Line Remove  Minimum Line Length
        'max_line_gap': 200,   # Long Line Remove Maximum Line Gap
    },

    'close': {
        'kernel_size_row': 25,  # Closing Kernel Size
        'kernel_size_col': 1,   # Closing Kernel Size
    },

    'contour': {
        'min_width': 20,  # Minimum Contour Rectangle Size
        'min_height': 20,  # Minimum Contour Rectangle Size
        'min_width_for_resize': 8,  # Minimum Contour Rectangle Size
        'min_height_for_resize': 8,  # Minimum Contour Rectangle Size
        'retrieve_mode': 3,    # RETR_EXTERNAL = 0. RETR_LIST = 1, RETR_CCOMP = 2, RETR_TREE = 3, RETR_FLOODFILL = 4
        'approx_method': 2,  # CHAIN_APPROX_NONE = 1, CHAIN_APPROX_SIMPLE = 2, CHAIN_APPROX_TC89_KCOS = 4, CHAIN_APPROX_TC89_L1 = 3
    }
}
