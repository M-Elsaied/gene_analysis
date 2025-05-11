# streamlit_gene_signature_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import gseapy as gp
from gseapy.plot import barplot

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🧬 Consistent Gene Signature Analysis from Drug Treatment")
st.markdown("""
Upload **multiple CSV files**, each corresponding to **one cell line** treated with the **same drug**. 
Ensure each file contains `ID_geneid`, `Name_GeneSymbol`, and `Value_LogDiffExp` columns.

🔹 Each file name should follow this format: `DrugName cell_line_name.xls - DrugName cell_line_name.xls.csv`  
🔹 One drug at a time only.
""")

# --- USER INPUT ---
log2fc_threshold = st.sidebar.slider("log₂ Fold Change Threshold", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
consistency_threshold = st.sidebar.slider("Cell Line Consistency Threshold (%)", min_value=50, max_value=100, value=70, step=5)

uploaded_files = st.file_uploader("Upload CSVs for one drug across different cell lines:", type="csv", accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        cell_line_data = {}

        for file in uploaded_files:
            file_path = os.path.join(tmpdir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

            # Parse cell line name from file name
            if " - " in file.name:
                cell_line = file.name.split(" - ")[0].replace("Camptothecin", "").replace(".xls", "").strip()
                df = pd.read_csv(file_path)
                cell_line_data[cell_line] = df

        st.success(f"Loaded {len(cell_line_data)} cell lines:")
        st.write(list(cell_line_data.keys()))

        # --- MERGE GENE EXPRESSION DATA ---
        gene_dfs = []
        for cell_line, df in cell_line_data.items():
            temp = df[['ID_geneid', 'Name_GeneSymbol', 'Value_LogDiffExp']].copy()
            temp = temp.rename(columns={'Value_LogDiffExp': cell_line})
            gene_dfs.append(temp)

        merged_df = reduce(lambda left, right: pd.merge(left, right, on=['ID_geneid', 'Name_GeneSymbol'], how='outer'), gene_dfs)
        merged_df.set_index(['ID_geneid', 'Name_GeneSymbol'], inplace=True)

        # --- FILTER FOR CONSISTENCY ---
        cell_line_count = len(cell_line_data)
        min_consistent = int(cell_line_count * (consistency_threshold / 100))

        upregulated_mask = (merged_df > log2fc_threshold).sum(axis=1)
        downregulated_mask = (merged_df < -log2fc_threshold).sum(axis=1)

        consistently_up_genes = upregulated_mask[upregulated_mask >= min_consistent].index
        consistently_down_genes = downregulated_mask[downregulated_mask >= min_consistent].index

        st.subheader("📊 Summary")
        st.write(f"**Total genes in matrix:** {merged_df.shape[0]}")
        st.write(f"**Genes upregulated in ≥{consistency_threshold}% cell lines:** {len(consistently_up_genes)}")
        st.write(f"**Genes downregulated in ≥{consistency_threshold}% cell lines:** {len(consistently_down_genes)}")

        # --- CLUSTERING HEATMAP ---
        consistent_genes = consistently_up_genes.union(consistently_down_genes)
        heatmap_data = merged_df.loc[consistent_genes].fillna(0)

        row_linkage = linkage(pdist(heatmap_data, metric='correlation'), method='average')
        col_linkage = linkage(pdist(heatmap_data.T, metric='correlation'), method='average')

        st.subheader("🧯 Hierarchical Clustering Heatmap")
        fig = sns.clustermap(
            heatmap_data,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            cmap='RdBu_r',
            center=0,
            linewidths=0.5,
            figsize=(10, 10)
        )
        st.pyplot(fig.fig)

        # --- ENRICHMENT ANALYSIS ---
        st.subheader("🔬 Pathway Enrichment (Enrichr via GSEApy)")

        def run_enrichment(gene_set, label):
            if len(gene_set) == 0:
                st.warning(f"No genes available for enrichment in {label} set.")
                return None

            symbols = [gene[1] for gene in gene_set]
            try:
                enr = gp.enrichr(
                    gene_list=symbols,
                    gene_sets=["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"],
                    organism='Human',
                    outdir=None,
                    cutoff=0.05
                )
                if enr.results.empty:
                    st.info(f"No significant enrichment found for {label}.")
                else:
                    st.write(f"**Top enriched pathways in {label} genes:**")
                    st.dataframe(enr.results.head(10))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    barplot(enr.results.sort_values('Adjusted P-value').head(10), title=f"{label} Enrichment", ax=ax)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in enrichment analysis for {label}: {e}")

        run_enrichment(consistently_up_genes, "Upregulated")
        run_enrichment(consistently_down_genes, "Downregulated")
