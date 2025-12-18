package com.example.demo.service.impl;



import org.springframework.stereotype.Service;

import com.example.demo.entity.AiToolDetail;
import com.example.demo.repository.AIToolDetailRepository;
import com.example.demo.service.AIToolDetailService;
import java.util.List;

@Service
public class AiToolDetailServiceImpl implements AIToolDetailService {

    private final AIToolDetailRepository detailRepository;

    public AiToolDetailServiceImpl(AIToolDetailRepository detailRepository) {
        this.detailRepository = detailRepository;
    }

    @Override
    public List<AiToolDetail> getDetailsByToolId(Long toolId) {
        return detailRepository.findByAiToolId(toolId);
    }

    @Override
    public AiToolDetail getDetailById(Long id) {
        return detailRepository.findById(id).orElse(null);
    }

    @Override
    public AiToolDetail saveDetail(AiToolDetail detail) {
        return detailRepository.save(detail);
    }
}

