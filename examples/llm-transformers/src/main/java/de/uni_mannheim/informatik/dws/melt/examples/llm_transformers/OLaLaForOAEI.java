package de.uni_mannheim.informatik.dws.melt.examples.llm_transformers;

import de.uni_mannheim.informatik.dws.melt.matching_base.IMatcher;
import de.uni_mannheim.informatik.dws.melt.matching_jena.MatcherPipelineYAAAJenaConstructor;
import de.uni_mannheim.informatik.dws.melt.matching_jena.MatcherYAAAJena;
import de.uni_mannheim.informatik.dws.melt.matching_jena.TextExtractor;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.elementlevel.HighPrecisionMatcher;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.filter.BadHostsFilter;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.filter.ConfidenceFilter;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.filter.extraction.NaiveDescendingExtractor;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.metalevel.AddAlignmentMatcher;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.metalevel.ConfidenceCombiner;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.StringProcessing;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorOnlyLabel;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorSet;
import de.uni_mannheim.informatik.dws.melt.matching_ml.python.nlptransformers.LLMBinaryFilter;
import de.uni_mannheim.informatik.dws.melt.matching_ml.python.nlptransformers.SentenceTransformersMatcher;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Alignment;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Correspondence;
import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Properties;
import org.apache.jena.ontology.OntModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 */
public class OLaLaForOAEI implements IMatcher<OntModel,Alignment,Properties> {
    private static final Logger LOGGER = LoggerFactory.getLogger(OLaLaForOAEI.class);
    
    private File transformersCache = null;
    private String gpus = "0";
    private int candidateLimit = 20;
    
    
    
    
    @Override
    public Alignment match(OntModel source, OntModel target, Alignment inputAlignment, Properties parameters) throws Exception {
        
        SentenceTransformersMatcher biEncoder = new SentenceTransformersMatcher(
            TextExtractor.appendStringPostProcessing(new TextExtractorSet(), StringProcessing::normalizeOnlyCamelCaseAndUnderscore),
            "multi-qa-mpnet-base-dot-v1"//"all-MiniLM-L6-v2"
        );
        biEncoder.setMultipleTextsToMultipleExamples(true);
        biEncoder.setTopK(5); 
        if(this.transformersCache != null)
            biEncoder.setTransformersCache(this.transformersCache);
        if(this.gpus != null)
            biEncoder.setCudaVisibleDevices(this.gpus);
        biEncoder.addResourceFilter(SentenceTransformersPredicateBadHosts.class);
        
        
        String model = "Qwen/Qwen2.5-7B-Instruct";
                
        LLMBinaryFilter llmTransformersFilter = new LLMBinaryFilter(
                new TextExtractorOnlyLabel(), 
                model,
                CLIOptions.PREDEFINED_PROMPTS.get(7));
        llmTransformersFilter.setMultipleTextsToMultipleExamples(true);
        if(this.transformersCache != null)
            llmTransformersFilter.setTransformersCache(this.transformersCache);
        if(this.gpus != null)
            llmTransformersFilter.setCudaVisibleDevices(this.gpus);
        
        llmTransformersFilter
                .addGenerationArgument("max_new_tokens", 10)
                .addGenerationArgument("temperature", 0.0);
        llmTransformersFilter
                .addLoadingArgument("device_map", "auto")
                .addLoadingArgument("torch_dtype", "bfloat16");
        
        MatcherPipelineYAAAJenaConstructor highPrecision = new MatcherPipelineYAAAJenaConstructor(
            new HighPrecisionMatcher(),
            new BadHostsFilter()
        );
        Alignment highPrecisionAlignment = highPrecision.match(source, target, inputAlignment, parameters);
        
        MatcherPipelineYAAAJenaConstructor matcher;
        if(this.candidateLimit > 0){
            matcher = new MatcherPipelineYAAAJenaConstructor(
                biEncoder,
                new CandidateLimitFilter(this.candidateLimit),
                llmTransformersFilter,
                new ConfidenceCombiner(LLMBinaryFilter.class),
                new AddAlignmentMatcher(highPrecisionAlignment),
                new NaiveDescendingExtractor(),
                new ConfidenceFilter(0.5)
            );
        }else{
            matcher = new MatcherPipelineYAAAJenaConstructor(
                biEncoder, 
                llmTransformersFilter,
                new ConfidenceCombiner(LLMBinaryFilter.class),
                new AddAlignmentMatcher(highPrecisionAlignment),
                new NaiveDescendingExtractor(),
                new ConfidenceFilter(0.5)
            );
        }
        
        return matcher.match(source, target, inputAlignment, parameters);
    }

    public File getTransformersCache() {
        return transformersCache;
    }

    public void setTransformersCache(File transformersCache) {
        this.transformersCache = transformersCache;
    }

    public String getGpus() {
        return gpus;
    }

    public void setGpus(String gpus) {
        this.gpus = gpus;
    }
    
    public int getCandidateLimit() {
        return candidateLimit;
    }

    public void setCandidateLimit(int candidateLimit) {
        this.candidateLimit = candidateLimit;
    }
    
    private static class CandidateLimitFilter extends MatcherYAAAJena {
        private final int limit;
        
        CandidateLimitFilter(int limit) {
            this.limit = limit;
        }
        
        @Override
        public Alignment match(OntModel source, OntModel target, Alignment inputAlignment, Properties properties) throws Exception {
            if(inputAlignment.size() <= this.limit){
                LOGGER.info("Candidate limit is {} and input alignment has {} correspondences. Keeping all candidates.", this.limit, inputAlignment.size());
                return inputAlignment;
            }
            
            List<Correspondence> ordered = new ArrayList<>(inputAlignment);
            ordered.sort(
                Comparator.comparingDouble(Correspondence::getConfidence).reversed()
                    .thenComparing(Correspondence::getEntityOne)
                    .thenComparing(Correspondence::getEntityTwo)
            );
            
            Alignment limitedAlignment = new Alignment(inputAlignment, false);
            limitedAlignment.addAll(ordered.subList(0, this.limit));
            LOGGER.info("Limited candidate alignment from {} to {} correspondences before LLM filtering.", inputAlignment.size(), limitedAlignment.size());
            return limitedAlignment;
        }
    }
}
