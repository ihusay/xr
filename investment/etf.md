# ETF 观察

## SOXX ETF 持仓和更新方式

iShares Semiconductor ETF（SOXX）跟踪 NYSE Semiconductor Index，用于观察美国上市半导体产业链。当前官方口径：数据截至 2026-06-18；基金规模约 $46.67B；持仓 30 只；费用率 0.34%。下表按官方 `Holdings` 工作表的 `Weight (%)` 排序，剔除现金、货币基金、保证金和期货行。

持仓占比截面时间：2026-06-18（官方下载文件 `Holdings` 工作表里的 `Fund Holdings as of: Jun 18, 2026`）。

| 底层公司 | 权重 | 说明 |
| --- | ---: | --- |
| Micron Technology | 8.39% | `MU`，存储 / DRAM / HBM。 |
| Advanced Micro Devices | 7.48% | `AMD`，CPU/GPU，云算力芯片。 |
| NVIDIA | 7.17% | `NVDA`，GPU / AI 加速器 / AI 系统。 |
| Broadcom | 6.61% | `AVGO`，AI ASIC、网络和连接芯片。 |
| Intel | 6.07% | `INTC`，CPU / 芯片制造。 |
| Marvell Technology | 5.44% | `MRVL`，定制 AI 芯片、网络和存储连接。 |
| Applied Materials | 4.92% | `AMAT`，半导体设备。 |
| KLA | 4.85% | `KLAC`，量测 / 检测设备。 |
| Lam Research | 4.39% | `LRCX`，半导体设备。 |
| TSMC | 4.12% | `TSM`，晶圆代工 / 先进制程。 |
| Texas Instruments | 3.79% | `TXN`，模拟 / 嵌入式。 |
| Analog Devices | 3.77% | `ADI`，模拟 / 混合信号。 |
| NXP Semiconductors | 3.50% | `NXPI`，汽车 / 工业 / 边缘芯片。 |
| Monolithic Power Systems | 3.31% | `MPWR`，电源管理。 |
| Qualcomm | 3.23% | `QCOM`，端侧 AI / 移动 SoC。 |
| Teradyne | 3.04% | `TER`，半导体测试 / 自动化测试。 |
| Astera Labs | 2.61% | `ALAB`，AI 数据中心互联 / CXL / PCIe。 |
| Microchip Technology | 2.37% | `MCHP`，MCU / 模拟 / 嵌入式。 |
| ASML | 2.23% | `ASML`，光刻设备。 |
| ON Semiconductor | 2.12% | `ON`，功率 / 汽车半导体。 |
| Credo Technology | 2.01% | `CRDO`，高速连接 / DSP / SerDes。 |
| Entegris | 1.21% | `ENTG`，半导体材料、过滤和化学品。 |
| MACOM Technology Solutions | 1.21% | `MTSI`，RF / 光通信 / 模拟。 |
| ASE Technology | 1.09% | `ASX`，封测。 |
| Arm Holdings | 0.98% | `ARM`，IP / 端侧 CPU 架构。 |
| United Microelectronics | 0.90% | `UMC`，成熟制程晶圆代工。 |
| STMicroelectronics | 0.83% | `STM`，MCU / 功率 / 汽车半导体。 |
| Nova | 0.82% | `NVMI`，量测设备。 |
| Rambus | 0.68% | `RMBS`，内存接口 / IP。 |
| Skyworks Solutions | 0.48% | `SWKS`，射频前端 / 无线芯片。 |

更新方式：

1. 打开 iShares 官方 SOXX 页面：`https://www.ishares.com/us/products/239705/ishares-semiconductor-etf`。
2. 点击页面里的 `Data Download`，或访问 BlackRock 官方下载接口：`https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appSubType=ISHARES&appType=PRODUCT_PAGE&component=fundDownload&locale=en_US&portfolioId=239705&targetSite=us-ishares&userType=individual`。
3. 读取下载文件里的 `Holdings` 工作表，使用 `Fund Holdings as of` 作为持仓日期。
4. 剔除 `USD CASH`、`BLK CSH FND TREASURY SL AGENCY`、`CASH COLLATERAL`、`ETD USD BALANCE`、期货等现金/抵押品/衍生品行。
5. 按 `Weight (%)` 从高到低排序；SOXX 通常不需要像 DRAM 那样合并 TRS/swap 暴露。

数据源：[iShares Semiconductor ETF - SOXX](https://www.ishares.com/us/products/239705/ishares-semiconductor-etf)、[iShares SOXX Data Download](https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appSubType=ISHARES&appType=PRODUCT_PAGE&component=fundDownload&locale=en_US&portfolioId=239705&targetSite=us-ishares&userType=individual)。

## DRAM ETF 持仓和更新方式

Roundhill Memory ETF（DRAM）用于观察全球存储产业链，当前口径按底层公司合并普通股和 total return swaps（TRS/swap）暴露；不把美债、货币基金、外汇和 `Cash&Other` 计入底层公司权重。

持仓占比截面时间：2026-06-18（按官方持仓文件名 `FilepointRoundhill.40RU.RU_Holdings_06182026.csv`；CSV 内部 `Date` 字段为 2026-06-22）。

| 底层公司 | 权重 | 说明 |
| --- | ---: | --- |
| Micron Technology | 27.57% | `MU` 普通股 + Micron TRS/swap 合并。 |
| SK Hynix | 26.87% | `000660.KS` 普通股 + SK Hynix TRS/swap 合并。 |
| Samsung Electronics | 17.64% | `005930.KS` 普通股 + Samsung TRS/swap 合并。 |
| Kioxia Holdings | 8.00% | `285A.T`，NAND/SSD。 |
| Sandisk | 5.52% | `SNDK`，NAND/SSD。 |
| Western Digital / WD | 4.36% | `WDC`，HDD/存储系统。 |
| Seagate Technology | 4.27% | `STX`，HDD/存储。 |
| Nanya Technology | 3.27% | `2408.TW`，DRAM。 |
| Winbond Electronics | 2.08% | `2344.TW`，利基 DRAM / Flash。 |

更新方式：

1. 打开 Roundhill 官方 DRAM 页面：`https://www.roundhillinvestments.com/etf/dram/`。
2. 官方页面会从 `https://www.roundhillinvestments.com/assets/data/FilepointRoundhill.40RU.RU_Holdings_MMDDYYYY.csv` 读取持仓文件；`MMDDYYYY` 按日期替换，通常从最近日期往前找可用文件。
3. 在 CSV 里筛选 `Account == DRAM`。
4. 剔除 `Cash&Other`、United States Treasury Bill、First American Government Obligations Fund、KRW、TWD 等现金/抵押品/外汇行。
5. 对同一底层公司合并普通股和 TRS/swap 行，再按 `Weightings` 求和排序。

数据源：[Roundhill Memory ETF - DRAM](https://www.roundhillinvestments.com/etf/dram/)、[Roundhill DRAM latest holdings CSV](https://www.roundhillinvestments.com/assets/data/FilepointRoundhill.40RU.RU_Holdings_06182026.csv)。
