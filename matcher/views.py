from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .nlp_engine import calculate_match_score

def home(request):
    return render(request, "matcher/index.html")

@csrf_exempt
def match_cv_jd(request):
    if request.method == "POST":
        data = json.loads(request.body)

        cv_text = data.get("cv_text", "")
        jd_text = data.get("jd_text", "")

        score, matched_words, matched_bigrams = calculate_match_score(cv_text, jd_text)

        return JsonResponse({
            "match_score": score,
            "matched_words": matched_words,
            "matched_bigrams": matched_bigrams
        })
