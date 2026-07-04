# ETF 观察

## SOXX ETF 持仓和更新方式

iShares Semiconductor ETF（SOXX）跟踪 NYSE Semiconductor Index，用于观察美国上市半导体产业链。当前官方口径：数据截至 2026-07-02；基金规模约 $42.13B；持仓 30 只；费用率 0.34%。下表按官方 `Holdings` 工作表的 `Weight (%)` 排序，剔除现金、货币基金、保证金和期货行。

持仓占比截面时间：2026-07-02（官方下载文件 `Holdings` 工作表里的 `Fund Holdings as of: Jul 02, 2026`）。

| 底层公司 | 权重 | 说明 |
| --- | ---: | --- |
| Micron Technology | 8.16% | `MU`，存储 / DRAM / HBM。 |
| Advanced Micro Devices | 8.15% | `AMD`，CPU/GPU，云算力芯片。 |
| NVIDIA | 7.50% | `NVDA`，GPU / AI 加速器 / AI 系统。 |
| Broadcom | 6.56% | `AVGO`，AI ASIC、网络和连接芯片。 |
| Intel | 6.17% | `INTC`，CPU / 芯片制造。 |
| Applied Materials | 5.44% | `AMAT`，半导体设备。 |
| KLA | 4.98% | `KLAC`，量测 / 检测设备。 |
| Marvell Technology | 4.86% | `MRVL`，定制 AI 芯片、网络和存储连接。 |
| Lam Research | 4.49% | `LRCX`，半导体设备。 |
| TSMC | 4.38% | `TSM`，晶圆代工 / 先进制程。 |
| Texas Instruments | 3.90% | `TXN`，模拟 / 嵌入式。 |
| Analog Devices | 3.70% | `ADI`，模拟 / 混合信号。 |
| NXP Semiconductors | 3.46% | `NXPI`，汽车 / 工业 / 边缘芯片。 |
| Monolithic Power Systems | 3.08% | `MPWR`，电源管理。 |
| Teradyne | 2.90% | `TER`，半导体测试 / 自动化测试。 |
| Astera Labs | 2.88% | `ALAB`，AI 数据中心互联 / CXL / PCIe。 |
| Qualcomm | 2.85% | `QCOM`，端侧 AI / 移动 SoC。 |
| ASML | 2.32% | `ASML`，光刻设备。 |
| Microchip Technology | 2.27% | `MCHP`，MCU / 模拟 / 嵌入式。 |
| Credo Technology | 2.02% | `CRDO`，高速连接 / DSP / SerDes。 |
| ON Semiconductor | 1.80% | `ON`，功率 / 汽车半导体。 |
| ASE Technology | 1.28% | `ASX`，封测。 |
| MACOM Technology Solutions | 1.12% | `MTSI`，RF / 光通信 / 模拟。 |
| Entegris | 1.12% | `ENTG`，半导体材料、过滤和化学品。 |
| United Microelectronics | 1.04% | `UMC`，成熟制程晶圆代工。 |
| STMicroelectronics | 0.81% | `STM`，MCU / 功率 / 汽车半导体。 |
| Arm Holdings | 0.79% | `ARM`，IP / 端侧 CPU 架构。 |
| Nova | 0.75% | `NVMI`，量测设备。 |
| Rambus | 0.61% | `RMBS`，内存接口 / IP。 |
| Skyworks Solutions | 0.47% | `SWKS`，射频前端 / 无线芯片。 |

更新方式：

1. 打开 iShares 官方 SOXX 页面：`https://www.ishares.com/us/products/239705/ishares-semiconductor-etf`。
2. 点击页面里的 `Data Download`，或访问 BlackRock 官方下载接口：`https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appSubType=ISHARES&appType=PRODUCT_PAGE&component=fundDownload&locale=en_US&portfolioId=239705&targetSite=us-ishares&userType=individual`。
3. 读取下载文件里的 `Holdings` 工作表，使用 `Fund Holdings as of` 作为持仓日期。
4. 剔除 `USD CASH`、`BLK CSH FND TREASURY SL AGENCY`、`CASH COLLATERAL`、`ETD USD BALANCE`、期货等现金/抵押品/衍生品行。
5. 按 `Weight (%)` 从高到低排序；SOXX 通常不需要像 DRAM 那样合并 TRS/swap 暴露。

数据源：[iShares Semiconductor ETF - SOXX](https://www.ishares.com/us/products/239705/ishares-semiconductor-etf)、[iShares SOXX Data Download](https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document?appSubType=ISHARES&appType=PRODUCT_PAGE&component=fundDownload&locale=en_US&portfolioId=239705&targetSite=us-ishares&userType=individual)。

## DRAM ETF 持仓和更新方式

Roundhill Memory ETF（DRAM）用于观察全球存储产业链，当前口径按底层公司合并普通股和 total return swaps（TRS/swap）暴露；不把美债、货币基金、外汇和 `Cash&Other` 计入底层公司权重。

持仓占比截面时间：2026-07-02（按官方持仓文件名 `FilepointRoundhill.40RU.RU_Holdings_07022026.csv`；CSV 内部 `Date` 字段为 2026-07-06）。

| 底层公司 | 权重 | 说明 |
| --- | ---: | --- |
| Micron Technology | 25.96% | `MU` 普通股 + Micron TRS/swap 合并。 |
| Samsung Electronics | 25.40% | `005930.KS` 普通股 + Samsung TRS/swap 合并。 |
| SK Hynix | 23.46% | `000660.KS` 普通股 + SK Hynix TRS/swap 合并。 |
| Sandisk | 4.75% | `SNDK`，NAND/SSD。 |
| Kioxia Holdings | 4.45% | `285A.T`，NAND/SSD。 |
| Western Digital / WD | 4.31% | `WDC`，HDD/存储系统。 |
| Seagate Technology | 4.24% | `STX`，HDD/存储。 |
| GigaDevice Semiconductor | 3.12% | `603986.SH`，存储 / MCU / NOR Flash。 |
| Nanya Technology | 1.94% | `2408.TW`，DRAM。 |
| Winbond Electronics | 1.17% | `2344.TW`，利基 DRAM / Flash。 |
| Phison Electronics | 0.70% | `8299.TW`，NAND 控制器 / SSD 方案。 |
| Macronix International | 0.39% | `2337.TW`，NOR Flash / ROM。 |

更新方式：

1. 打开 Roundhill 官方 DRAM 页面：`https://www.roundhillinvestments.com/etf/dram/`。
2. 官方页面会从 `https://www.roundhillinvestments.com/assets/data/FilepointRoundhill.40RU.RU_Holdings_MMDDYYYY.csv` 读取持仓文件；`MMDDYYYY` 按日期替换，通常从最近日期往前找可用文件。
3. 在 CSV 里筛选 `Account == DRAM`。
4. 剔除 `Cash&Other`、United States Treasury Bill、First American Government Obligations Fund、KRW、TWD 等现金/抵押品/外汇行。
5. 对同一底层公司合并普通股和 TRS/swap 行，再按 `Weightings` 求和排序。

数据源：[Roundhill Memory ETF - DRAM](https://www.roundhillinvestments.com/etf/dram/)、[Roundhill DRAM latest holdings CSV](https://www.roundhillinvestments.com/assets/data/FilepointRoundhill.40RU.RU_Holdings_07022026.csv)。
