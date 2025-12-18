package com.example.demo.service.impl;

import com.example.demo.entity.ContactMessage;
import com.example.demo.repository.ContactMessageRepository;
import com.example.demo.service.ContactMessageService;
import java.util.List;

import org.springframework.stereotype.Service;

@Service
public class ContactMessageServiceImpl implements ContactMessageService {

    private final ContactMessageRepository repository;

    public ContactMessageServiceImpl(ContactMessageRepository repository) {
        this.repository = repository;
    }

  

    @Override
    public List<ContactMessage> getAllMessages() {
        return repository.findAll();
    }

	@Override
	public ContactMessage saveMessage(ContactMessage message) {
		// TODO Auto-generated method stub
		return repository.save(message);
	}


}
