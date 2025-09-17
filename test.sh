SUBSET_LIST=(ImageNet-1K N24News HatefulMemes VOC2007 SUN397 A-OKVQA MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 OK-VQA RefCOCO DocVQA InfographicsVQA ChartQA NIGHTS FashionIQ ScienceQA Visual7W VizWiz GQA TextVQA CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t Wiki-SS-NQ WebQA OVEN EDIS RefCOCO-Matching Visual7W-Pointing VisDial)

for f in "${SUBSET_LIST[@]}"; do
    echo $f
done