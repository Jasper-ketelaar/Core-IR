package nl.tudelft.coreir;

import io.anserini.ltr.feature.BM25Stat;
import io.anserini.ltr.AvgPooler;
import io.anserini.ltr.FeatureExtractorUtils;


public class L2R {
//    put all the features in a list
    private BM25Stat bm25;
    private FeatureExtractorUtils featureExtractorUtils;


    public L2R(){
        AvgPooler avgPooler = new AvgPooler();
        float k1 = 3.44;
        float b = 0.87;
        this.bm25 = new BM25Stat(avgPooler, k1, b);
        this.featureExtractorUtils = new FeatureExtractorUtils();
    }

    public FeatureExtractorUtils addFeatures(){
        FeatureExtractorUtils featureExtractorUtils.add(bm25);

        return featureExtractorUtils;
    }
}