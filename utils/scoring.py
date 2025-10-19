def compute_score(delay_prob, co2_kg, duration_hr, weights=(0.4, 0.4, 0.2)):
    w1, w2, w3 = weights
    # Normalize inversely (lower delay & CO₂ are better)
    delay_score = 1 - min(delay_prob, 1)
    co2_score = 1 / (1 + co2_kg / 200)  # simple normalization
    duration_score = 1 / (1 + duration_hr / 10)
    total = w1*delay_score + w2*co2_score + w3*duration_score
    return round(total, 3)
