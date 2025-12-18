package com.example.demo.service.impl;

import com.example.demo.repository.OptionRepository;
import com.example.demo.entity.Option;
import com.example.demo.service.OptionService;

import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class OptionServiceImpl implements OptionService {

    private final OptionRepository optionRepository;

    public OptionServiceImpl(OptionRepository optionRepository) {
        this.optionRepository = optionRepository;
    }

    @Override
    public List<Option> getOptionsByQuestion(Long questionId) {
        return optionRepository.findByQuestionId(questionId);
    }
}
