def calculate_final_verdict(resnet_score, siglip_score, sdxl_score, deepfake_score):
    """
    Averages the four AI model scores and returns a final string ("REAL" or "FAKE") 
    alongside the combined percentage confidence.
    """
    average_score = (resnet_score + siglip_score + sdxl_score + deepfake_score) / 4.0
    
    # If average "fake" score is > 0.5, it's considered FAKE
    if average_score > 0.5:
        verdict = "FAKE"
        confidence = average_score * 100
    else:
        verdict = "REAL"
        confidence = (1.0 - average_score) * 100
        
    return verdict, confidence
