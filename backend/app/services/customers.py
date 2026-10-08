from app.core.db import get_connection


def list_customers(
    risk_level=None, contract=None, min_tenure=None, max_tenure=None,
    min_monthly=None, max_monthly=None, internet_service=None,
    search=None, sort_by="churn_probability", sort_dir="desc",
    page=1, page_size=25,
):
    conn = get_connection()
    where = []
    params = []
    if risk_level:
        placeholders = ",".join("?" * len(risk_level))
        where.append(f"risk_level IN ({placeholders})")
        params.extend(risk_level)
    if contract:
        where.append("Contract = ?")
        params.append(contract)
    if min_tenure is not None:
        where.append("tenure >= ?")
        params.append(min_tenure)
    if max_tenure is not None:
        where.append("tenure <= ?")
        params.append(max_tenure)
    if min_monthly is not None:
        where.append("MonthlyCharges >= ?")
        params.append(min_monthly)
    if max_monthly is not None:
        where.append("MonthlyCharges <= ?")
        params.append(max_monthly)
    if internet_service:
        where.append("InternetService = ?")
        params.append(internet_service)
    if search:
        where.append("customerID LIKE ?")
        params.append(f"%{search}%")

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    allowed_sort = {
        "churn_probability", "CLV", "tenure", "MonthlyCharges",
        "churn_risk_6m", "customerID",
    }
    if sort_by not in allowed_sort:
        sort_by = "churn_probability"
    sort_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    count_row = conn.execute(f"SELECT COUNT(*) as c FROM customers {where_clause}", params).fetchone()
    total = count_row["c"]

    offset = (page - 1) * page_size
    query = f"""
        SELECT customerID, Contract, tenure, MonthlyCharges, TotalCharges,
               churn_probability, risk_level, CLV, churn_risk_6m,
               primary_risk_driver, InternetService, PaymentMethod, Churn
        FROM customers
        {where_clause}
        ORDER BY {sort_by} {sort_dir}
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(query, params + [page_size, offset]).fetchall()
    conn.close()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "customers": [dict(r) for r in rows],
    }


def get_customer(customer_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM customers WHERE customerID = ?", (customer_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def dashboard_summary():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) c FROM customers").fetchone()["c"]
    high_risk = conn.execute(
        "SELECT COUNT(*) c FROM customers WHERE risk_level IN ('High','Critical')"
    ).fetchone()["c"]
    avg_prob = conn.execute("SELECT AVG(churn_probability) a FROM customers").fetchone()["a"]
    revenue_at_risk = conn.execute(
        "SELECT SUM(CLV) s FROM customers WHERE risk_level IN ('High','Critical')"
    ).fetchone()["s"]
    total_clv = conn.execute("SELECT SUM(CLV) s FROM customers").fetchone()["s"]
    by_segment = conn.execute(
        "SELECT InternetService as segment, SUM(CLV) as revenue_at_risk, COUNT(*) as customers "
        "FROM customers WHERE risk_level IN ('High','Critical') GROUP BY InternetService"
    ).fetchall()
    by_contract = conn.execute(
        "SELECT Contract, COUNT(*) as n, AVG(churn_probability) as avg_risk FROM customers GROUP BY Contract"
    ).fetchall()
    risk_distribution = conn.execute(
        "SELECT risk_level, COUNT(*) as n FROM customers GROUP BY risk_level"
    ).fetchall()
    top_opportunities = conn.execute(
        "SELECT customerID, churn_probability, risk_level, CLV, churn_risk_6m, "
        "primary_risk_driver FROM customers WHERE risk_level IN ('High','Critical') "
        "ORDER BY CLV DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return {
        "total_customers": total,
        "high_risk_customers": high_risk,
        "predicted_churn_rate": round(avg_prob, 4) if avg_prob else None,
        "revenue_at_risk": round(revenue_at_risk, 2) if revenue_at_risk else 0,
        "total_portfolio_clv": round(total_clv, 2) if total_clv else 0,
        "revenue_at_risk_by_segment": [dict(r) for r in by_segment],
        "risk_by_contract": [dict(r) for r in by_contract],
        "risk_distribution": [dict(r) for r in risk_distribution],
        "top_retention_opportunities": [dict(r) for r in top_opportunities],
    }
