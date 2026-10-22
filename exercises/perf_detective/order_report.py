"""주문 리포트 생성기 (느린 프로그램). 실행: python order_report.py
고객 수, 상품 수, 총 매출, 상위 상품 5개를 계산해 report.txt 로 저장한다.
목표: 왜 느린지 가설을 세우고, 측정으로 확인해서, 결과가 같도록 빠르게 만든다."""
import random
import time


def make_orders(n, seed=7):
    rng = random.Random(seed)
    return [
        {"customer": rng.randrange(1, 4001), "product": rng.randrange(1, 801),
         "price": round(rng.uniform(1, 200), 2), "qty": rng.randrange(1, 6)}
        for _ in range(n)
    ]


def unique_customers(orders):
    seen = []
    for o in orders:
        if o["customer"] not in seen:
            seen.append(o["customer"])
    return seen


def spend_per_customer(orders, customers):
    result = {}
    for c in customers:
        total = 0.0
        for o in orders:
            if o["customer"] == c:
                total += o["price"] * o["qty"]
        result[c] = round(total, 2)
    return result


def top_products(orders, k=5):
    counts = {}
    for o in orders:
        counts[o["product"]] = counts.get(o["product"], 0) + o["qty"]
        ranking = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranking[:k]


def build_report(customers, spend, top):
    text = ""
    for c in customers:
        text += f"customer {c}: {spend[c]}\n"
    text += "top products: " + ", ".join(f"{p}:{q}" for p, q in top) + "\n"
    return text


def main():
    t0 = time.time()
    orders = make_orders(30000)
    customers = unique_customers(orders)
    spend = spend_per_customer(orders, customers)
    top = top_products(orders)
    report = build_report(customers, spend, top)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"고객 {len(customers)}명, 총매출 {sum(spend.values()):.2f}, 상위상품 {top[0]}")
    print(f"걸린 시간 {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
