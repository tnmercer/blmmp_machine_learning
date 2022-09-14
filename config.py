# ==================================================================================================
# global variables
# ==================================================================================================

DEBUG = True


# ==================================================================================================
# application variables
# ==================================================================================================

# Portfolio 2: Vanguard Total Stock Market ETF
vanguard_total_stock_etf = {
    'VTI': 100
}

# Portfolios 179-183: Fidelity Index Focused Models
fidelity_index_focused_models = {
    'FXAIX': 35,  
    'FSMDX': 3,
    'FSSNX': 4,
    'FSGGX': 18,
    'FXNAX': 35,
    'FCONX':3,
    # Replaced SPRXX with BIL as no data on yahoo finanne
    'BIL': 2
}

# Rick Ferri's Core 4
rick_ferris_core = {
    'VTSAX': 48,  
    'VTIAX': 24,
    'VNQ': 8,
    'VBTLX': 20  
}
    
# Betterment Portfolios
betterment = {
    'VFIAX': 15, 
    'VVIAX': 15, 
    'VTMGX': 15, 
    'VEMAX': 6, 
    'VIMAX': 5, 
    'VSMAX': 4, 
    'VIPSX': 20, 
    'VSBSX': 20 
}

all_portfolios = {
	'Vanguard_Total_Stock': vanguard_total_stock_etf,        # 100% stock fund
	'Betterment_Portfolio': betterment,                      # 60/40 fund
	'Fidelity_Portfolio':   fidelity_index_focused_models,   # 60/40 fund
	'Rick_Ferris_Core':     rick_ferris_core,                # 80/20 fund
}

# Ticker to name library
ticker_library = {
    'VTI': 'Vanguard Total Stock Market ETF',
    'FXAIX': 'Fidelity 500 Index Fund',
    'FSMDX': 'Fidelity Mid Cap Index Fund',
    'FSSNX': 'Fidelity Small Cap Index Fund',
    'FSGGX': 'Fidelity Ex-US Global Index Fund',
    'FXNAX': 'Fidelity US Bond Index Fund', 
    'FCONX': 'Fidelity Conservative US Bond Fund', 
    'SPRXX': 'Fidelity Core Money Market Fund',
    'VTSAX': 'Vanguard Total Stock Market Fund',
    'VTIAX': 'Vanguard Total International Stock Market Fund',
    'VGSLX': 'Vanguard REIT Index Fund',
    'VFIAX': 'Vanguard US Total Stock Market Index Fund',
    'VVIAX': 'Vanguard Value Index Fund',
    'VTMGX': 'Vanguard Developed Markets Index Fund',
    'VEMAX': 'Vanguard Emerging Markets Index Fund',
    'VIMAX': 'Vanguard Mid Cap Index Fund',
    'VSMAX': 'Vanguard Small Cap Value Index Fund',
    'VIPSX': 'Vanguard Inflation-Protected Securities Fund',
    'VSBSX': 'Vanguard Short Term Treasury Index Fund',
    'VBTLX': 'Vanguard Total Bond Market Fund',
	'BIL':   'BIL',
	'VNQ':   'VNQ',
}


# ==================================================================================================
# application settings
# ==================================================================================================

PLOT_WIDTH = 1200
PLOT_HEIGHT = 1000

