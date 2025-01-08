[Dataset](https://databento.com/portal/datasets/XNAS.ITCH)

## Dataset

- dataset is Nasdaq TotalView-ITCH, with dataset ID of `XNAS.ITCH`
- schema is `mbp-10`
- date in the paper is from 2017-01-01 to 2019-12-31. But `XNAS.ITCH` earliest available start date is 2018-05-01
- we only need about 1 week/1 month of data. I choose **1 week** of data, starting from 2024-12-01 to 2024-12-08
- we account for these symbols (products) only: AAPL, MSFT, NVDA, AMGN, GILD, TSLA, PEP, JPM, V, XOM

### Fetch Dataset

```python
dataset="XNAS.ITCH",
start="2024-12-01",
end="2024-12-08",
symbols=["AAPL", "MSFT", "NVDA", "AMGN", "GILD", "TSLA", "PEP", "JPM", "V", "XOM"],
schema="mbp-10",
split_duration="day",
```

```sh
drwxr-xr-x@     - 80026129  5 Jan 00:25 ├──  XNAS-20250105-S6R97734QU
.rw-r--r--@   602 80026129  5 Jan 00:25 │   ├──  condition.json
.rw-r--r--@  3.9k 80026129  5 Jan 00:25 │   ├──  manifest.json
.rw-r--r--@   863 80026129  5 Jan 00:25 │   ├──  metadata.json
.rw-r--r--@  2.0k 80026129  5 Jan 00:25 │   ├──  symbology.json
.rw-r--r--@  226M 80026129  5 Jan 00:26 │   ├──  xnas-itch-20241202.mbp-10.dbn.zst
.rw-r--r--@  199M 80026129  5 Jan 00:26 │   ├──  xnas-itch-20241203.mbp-10.dbn.zst
.rw-r--r--@  213M 80026129  5 Jan 00:25 │   ├──  xnas-itch-20241204.mbp-10.dbn.zst
.rw-r--r--@  224M 80026129  5 Jan 00:26 │   ├──  xnas-itch-20241205.mbp-10.dbn.zst
.rw-r--r--@  228M 80026129  5 Jan 00:26 │   └──  xnas-itch-20241206.mbp-10.dbn.zst
```

### EDA

- Every symbol has its own unique instrument id and publisher id
- The only actions that exist in the dataset are `A`dd, `F`ill, `T`rade, `C`ancel.
- For orders with `A` and `C` actions, the order book is updated immediately with the coming order
- For orders with `F` and `T` actions, the order book is not updated immediately. The change is either reflected in a order with the same action type right after that, or reflected through 1 or more `C` orders to fulfil the trade/fill.
- `F` action is for `B` orders, `T` action is for `A` orders

## Data Preprocessing

- For each day, we use each non-overlapping 30-minute window as a training sample for the linear regression models (self-impact and cross-impact)
- Within each window, OFIs and returns are calculated for every minute
- Each window will have its own linear regression model coefficients, and we will average the coefficients of all windows to get the final coefficients
