import akshare as ak

if __name__ == '__main__':
    # stock_sse_summary_df = ak.stock_sse_summary()
    # print(stock_sse_summary_df)
    #
    # stock_szse_sector_summary_df = ak.stock_szse_sector_summary(symbol="当年", date="202501")
    # print(stock_szse_sector_summary_df)

    # stock_rank_forecast_cninfo_df = ak.stock_rank_forecast_cninfo("20250520")
    # print(stock_rank_forecast_cninfo_df)

    # newsem = ak.stock_news_em(symbol="603777")
    # print(newsem)

    # fund_lof_spot_em = ak.fund_lof_spot_em()
    # print(fund_lof_spot_em)

    # futures_stock_shfe_js_df = ak.futures_stock_shfe_js(date="20250520")
    # print(futures_stock_shfe_js_df)
    #
    # article_epu_index_df = ak.article_epu_index(symbol="China")
    # print(article_epu_index_df)

    # stock_individual_info_em_df = ak.stock_individual_info_em(symbol="")
    # print(stock_individual_info_em_df)

    # stock_rank_ljqs_ths_df = ak.stock_rank_ljqs_ths()
    # print(stock_rank_ljqs_ths_df)

    # article_ff_crr_df = ak.article_ff_crr()
    # print(article_ff_crr_df)
    # # 将数据写入excel
    # article_ff_crr_df.to_excel("article_ff_crr.xlsx", index=False)

    # stock_hot_follow_xq_df = ak.stock_hot_keyword_em(symbol="SZ603228")
    # print(stock_hot_follow_xq_df)

    stock_zh_a_hist_df = ak.stock_zh_a_hist(
        symbol="600734",
        period="daily",
        start_date="20250101",
        end_date="20250713",
        adjust="hfq"
    )
    print(stock_zh_a_hist_df)
