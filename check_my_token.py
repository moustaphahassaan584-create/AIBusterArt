import os

# بنحاول نقرأ التوكن من "الخزنة" (GitHub Secrets)
token = os.getenv("REPLICATE_API_TOKEN")

if token:
    # لو التوكن موجود، هيطبع أول حرفين منه للأمان
    print("✅ Success! Your GitHub Secret is connected.")
    print(f"Token starts with: {token[:2]}****")
else:
    # لو التوكن مش موجود أو الاسم غلط
    print("❌ Error: Cannot find REPLICATE_API_TOKEN in Secrets.")
