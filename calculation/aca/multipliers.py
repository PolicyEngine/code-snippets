import pandas as pd

enrollment = {
    "Alabama": (202_847, 396_750),
    "Alaska": (21_251, 25_126),
    "Arizona": (181_741, 349_847),
    "Arkansas": (77_748, 134_543),
    "California": (1_701_375, 1_795_695),
    "Colorado": (180_309, 229_679),
    "Connecticut": (102_608, 131_600),
    "Delaware": (29_872, 43_129),
    "District of Columbia": (13_655, 11_484),
    "Florida": (2_585_959, 4_163_154),
    "Georgia": (638_596, 1_259_957),
    "Hawaii": (19_812, 20_486),
    "Idaho": (68_792, 102_396),
    "Illinois": (288_176, 378_297),
    "Indiana": (144_519, 295_264),
    "Iowa": (68_290, 113_095),
    "Kansas": (98_753, 168_080),
    "Kentucky": (61_618, 74_927),
    "Louisiana": (93_690, 229_588),
    "Maine": (59_234, 59_136),
    "Maryland": (168_097, 208_970),
    "Massachusetts": (224_613, 310_491),
    "Michigan": (276_655, 430_037),
    "Minnesota": (110_310, 132_544),
    "Mississippi": (126_731, 281_548),
    "Missouri": (217_196, 356_574),
    "Montana": (48_626, 65_991),
    "Nebraska": (88_875, 115_949),
    "Nevada": (90_397, 92_949),
    "New Hampshire": (48_316, 62_004),
    "New Jersey": (297_159, 405_257),
    "New Mexico": (34_678, 58_196),
    "New York": (210_523, 226_331),
    "North Carolina": (628_083, 928_561),
    "North Dakota": (29_185, 37_305),
    "Ohio": (235_976, 486_216),
    "Oklahoma": (171_473, 269_554),
    "Oregon": (130_283, 132_020),
    "Pennsylvania": (338_575, 424_411),
    "Rhode Island": (30_790, 41_346),
    "South Carolina": (281_301, 543_454),
    "South Dakota": (39_452, 49_533),
    "Tennessee": (254_118, 545_586),
    "Texas": (1_718_549, 3_383_171),
    "Utah": (252_336, 372_832),
    "Vermont": (24_588, 29_819),
    "Virginia": (287_063, 355_176),
    "Washington": (223_209, 280_820),
    "West Virginia": (21_108, 52_741),
    "Wisconsin": (196_848, 267_285),
    "Wyoming": (33_071, 39_943),
}

avg_aptc = {
    "Alabama": (710, 656),
    "Alaska": (692, 865),
    "Arizona": (446, 452),
    "Arkansas": (449, 476),
    "California": (459, 526),
    "Colorado": (368, 455),
    "Connecticut": (654, 766),
    "Delaware": (604, 585),
    "District of Columbia": (407, 561),
    "Florida": (552, 568),
    "Georgia": (464, 531),
    "Hawaii": (576, 544),
    "Idaho": (460, 395),
    "Illinois": (506, 545),
    "Indiana": (458, 452),
    "Iowa": (595, 507),
    "Kansas": (530, 561),
    "Kentucky": (481, 497),
    "Louisiana": (655, 647),
    "Maine": (470, 564),
    "Maryland": (383, 388),
    "Massachusetts": (373, 385),
    "Michigan": (390, 426),
    "Minnesota": (340, 351),
    "Mississippi": (566, 592),
    "Missouri": (546, 594),
    "Montana": (500, 504),
    "Nebraska": (615, 580),
    "Nevada": (435, 438),  # 2022: HHS/ASPE (KFF reported NR)
    "New Hampshire": (335, 350),
    "New Jersey": (504, 521),
    "New Mexico": (534, 551),
    "New York": (364, 455),
    "North Carolina": (583, 558),
    "North Dakota": (436, 433),
    "Ohio": (479, 498),
    "Oklahoma": (577, 575),
    "Oregon": (503, 524),
    "Pennsylvania": (523, 530),
    "Rhode Island": (427, 454),
    "South Carolina": (566, 553),
    "South Dakota": (649, 611),
    "Tennessee": (572, 580),
    "Texas": (539, 536),
    "Utah": (385, 421),
    "Vermont": (620, 702),
    "Virginia": (407, 405),
    "Washington": (438, 453),
    "West Virginia": (1057, 1035),
    "Wisconsin": (562, 572),
    "Wyoming": (873, 863),
}

rows = []
for state in enrollment:
    e22, e24 = enrollment[state]
    a22, a24 = avg_aptc[state]
    rows.append({
        "state": state,
        "enroll_2022": e22,
        "enroll_2024": e24,
        "vol_mult": e24 / e22,
        "aptc_2022": a22,
        "aptc_2024": a24,
        "val_mult": a24 / a22,
    })

df = pd.DataFrame(rows)

pd.set_option("display.max_rows", 60)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.3f}".format)

csv_path = "aca_ptc_multipliers_2022_2024.csv"
df.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}\n")
print(df.to_string(index=False))
print(f"\n{len(df)} jurisdictions")
print(f"\nVolume multiplier:  mean={df['vol_mult'].mean():.3f}  "
      f"median={df['vol_mult'].median():.3f}  "
      f"min={df['vol_mult'].min():.3f} ({df.loc[df['vol_mult'].idxmin(), 'state']})  "
      f"max={df['vol_mult'].max():.3f} ({df.loc[df['vol_mult'].idxmax(), 'state']})")
print(f"Value multiplier:   mean={df['val_mult'].mean():.3f}  "
      f"median={df['val_mult'].median():.3f}  "
      f"min={df['val_mult'].min():.3f} ({df.loc[df['val_mult'].idxmin(), 'state']})  "
      f"max={df['val_mult'].max():.3f} ({df.loc[df['val_mult'].idxmax(), 'state']})")
