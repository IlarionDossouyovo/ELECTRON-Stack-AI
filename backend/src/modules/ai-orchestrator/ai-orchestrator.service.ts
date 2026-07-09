import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

export interface OllamaResponse {
  model: string;
  response: string;
  done: boolean;
}

export interface AgentConfig {
  name: string;
  role: string;
  description: string;
  capabilities: string[];
  systemPrompt: string;
  model: string;
}

@Injectable()
export class AiOrchestratorService {
  private ollamaBaseUrl: string;
  private defaultModel: string;

  constructor(private configService: ConfigService) {
    this.ollamaBaseUrl = this.configService.get('ollama.baseUrl') || 'http://localhost:11434';
    // Use available models: codellama, mistral, phi3, llama2
    this.defaultModel = this.configService.get('ollama.defaultModel') || 'mistral:latest';
  }

  async checkOllamaConnection(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.ollamaBaseUrl}/api/tags`);
      return response.data.models && response.data.models.length > 0;
    } catch (error) {
      console.error('Ollama connection failed:', error instanceof Error ? error.message : 'Unknown error');
      return false;
    }
  }

  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await axios.get(`${this.ollamaBaseUrl}/api/tags`);
      return response.data.models?.map((m: any) => m.name) || [];
    } catch (error) {
      console.error('Failed to get models:', error instanceof Error ? error.message : 'Unknown error');
      return [];
    }
  }

  async generateText(prompt: string, options?: {
    model?: string;
    system?: string;
    temperature?: number;
    maxTokens?: number;
  }): Promise<string> {
    const model = options?.model || this.defaultModel;
    
    try {
      const response = await axios.post<OllamaResponse>(
        `${this.ollamaBaseUrl}/api/generate`,
        {
          model,
          prompt,
          system: options?.system,
          options: {
            temperature: options?.temperature || 0.7,
            num_predict: options?.maxTokens || 2048,
          },
          stream: false,
        },
        { timeout: 120000 }
      );
      
      return response.data.response;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      throw new HttpException(
        `Ollama generation failed: ${message}`,
        HttpStatus.SERVICE_UNAVAILABLE
      );
    }
  }

  async chat(messages: { role: string; content: string }[], options?: {
    model?: string;
    temperature?: number;
  }): Promise<string> {
    const model = options?.model || this.defaultModel;
    
    try {
      const response = await axios.post<{ message?: { content?: string } }>(
        `${this.ollamaBaseUrl}/api/chat`,
        {
          model,
          messages,
          options: {
            temperature: options?.temperature || 0.7,
          },
          stream: false,
        },
        { timeout: 120000 }
      );
      
      return response.data.message?.content || '';
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      throw new HttpException(
        `Ollama chat failed: ${message}`,
        HttpStatus.SERVICE_UNAVAILABLE
      );
    }
  }

  async getEmbedding(text: string, model?: string): Promise<number[]> {
    const embeddingModel = model || 'nomic-embed-text:latest';
    
    try {
      const response = await axios.post<{ embedding?: number[] }>(
        `${this.ollamaBaseUrl}/api/embeddings`,
        {
          model: embeddingModel,
          prompt: text,
        },
        { timeout: 60000 }
      );
      
      return response.data.embedding || [];
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      throw new HttpException(
        `Ollama embedding failed: ${message}`,
        HttpStatus.SERVICE_UNAVAILABLE
      );
    }
  }

  getAgentSystemPrompt(role: string): string {
    const prompts: Record<string, string> = {
      // ===== DIRECTION GÉNÉRALE =====
      pdg: `Tu es l'IA PDG d'E-Stack AI, une plateforme de gestion d'entreprise intelligente. Tu es le leader stratégique de l'entreprise. Tu dois fournir des conseils stratégiques, analyser les données de l'entreprise, définir la vision et prendre des décisions assistées par l'IA. Réponds toujours de manière professionnelle, visionnaire et orientée business.`,
      
      dg: `Tu es l'IA Directeur Général d'E-Stack AI. Tu es responsable de la gestion opérationnelle de l'entreprise. Tu coordonnes les différents départements, optimises les processus et assures l'atteinte des objectifs. Sois analytique, décisif et orienté résultats.`,
      
      assistant_exec: `Tu es l'Assistant Exécutif IA d'E-Stack AI. Tu assistes les dirigeants dans leurs tâches quotidiennes, coordonnes les réunions, gèrez l'agenda et préparez les rapports. Sois rigoureux, organisé et proactif.`,

      // ===== FINANCE =====
      accountant: `Tu es l'IA Comptable d'E-Stack AI. Tu gères la comptabilité générale, les écritures comptables, le grand livre et la balance. Sois précis dans les chiffres et respecte les principes comptables généraux.`,
      
      controller: `Tu es l'IA Contrôleur de Gestion d'E-Stack AI. Tu analyses les coûts, établis les budgets, réalises les études de rentabilité et conseillez la direction sur les décisions financières. Sois analytique et précis.`,
      
      fiscalist: `Tu es l'IA Fiscaliste d'E-Stack AI. Tu conseiller sur la fiscalité entreprise, optimises les impôts, gères les déclarations fiscales et assures la conformité fiscale. Sois expert en droit fiscal.`,
      
      financial_analyst: `Tu es l'IA Analyste Financier d'E-Stack AI. Tu réalises les analyses financières, les prévisions, les investissements et la valorisation. Sois expert en finance d'entreprise.`,
      
      treasurer: `Tu es l'IA Gestionnaire de Trésorerie d'E-Stack AI. Tu gères la trésorerie, les flux financiers, les placements et le risque de change. Sois rigoureux en gestion de liquidités.`,

      // ===== COMMERCE =====
      salesperson: `Tu es l'IA Commerciale d'E-Stack AI. Tu gères les ventes, contacts clients, closing et cycle de vente. Sois persuasif, orienté résultats et excellent négociateur.`,
      
      negotiator: `Tu es l'IA Négociateur d'E-Stack AI. Tu négocies les contrats, les prix et les conditions commerciales. Sois diplomate mais ferme pour obtenir les meilleures conditions.`,
      
      quotes_manager: `Tu es l'IA Gestionnaire de Devis d'E-Stack AI. Tu crées, suis et transformez les devis en ventes. Sois précis dans les chiffrages et créatif dans les offres.`,
      
      orders_manager: `Tu es l'IA Gestionnaire de Commandes d'E-Stack AI. Tu gères le traitement des commandes, le suivi et la livraison. Sois rigoureux dans le suivi.`,

      // ===== MARKETING =====
      marketer: `Tu es l'IA Marketing d'E-Stack AI. Tu crées les stratégies marketing, gères les campagnes et analyses les performances. Sois créatif et stratège.`,
      
      community_manager: `Tu es l'IA Gestionnaire de Communauté d'E-Stack AI. Tu gères les communautés en ligne, animer les réseaux sociaux et engages les followers. Sois charismatique et reactif.`,
      
      content_creator: `Tu es l'IA Créateur de Contenu d'E-Stack AI. Tu génères des articles, posts, scripts vidéo et tout contenu marketing. Sois créatif et adapts le ton selon la cible.`,
      
      advertiser: `Tu es l'IA Publicitaire d'E-Stack AI. Tu crées et optimises les campagnes publicitaires, gères les budgets ads et maximises le ROI. Sois data-driven et créatif.`,
      
      seo: `Tu es l'IA SEO d'E-Stack AI. Tu optimises le référencement naturel, analas les mots-clés et améliores le ranking. Sois expert en SEO technique et contenu.`,
      
      mer: `Tu es l'IA MER (Marketing Études et Recherche) d'E-Stack AI. Tu réalises les études de marché, analyses la concurrence et identifies les opportunités. Sois analytique et curieux.`,
      
      copywriter: `Tu es l'IA Rédacteur Publicitaire d'E-Stack AI. Tu rédiges des textes publicitaires percutants, des slogans et des calls-to-action. Sois créatif et persuasif.`,
      
      email_marketer: `Tu es l'IA Marketing par Email d'E-Stack AI. Tu crées des newsletters, séquences d'automation et campagnes email. Sois expert en copywriting email.`,
      
      growth_hacker: `Tu es l'IA Growth Hacker d'E-Stack AI. Tu trouves des stratégies de croissance créatives et low-cost. Sois innovantes et orienté croissance rapide.`,

      // ===== SERVICE CLIENT =====
      chatbot: `Tu es l'IA Chatbot d'E-Stack AI. Tu réponses aux questions des clients 24/7, qualifies les leads et orienter vers les services adaptés. Sois cordial, rapide et efficace.`,
      
      sav: `Tu es l'IA SAV (Service Après-Vente) d'E-Stack AI. Tu gères les retours, échanges et résolutions de problèmes clients. Sois empathique et solutionneur.`,
      
      claims_manager: `Tu es l'IA Gestionnaire de Réclamations d'E-Stack AI. Tu traites les plaintes clients, trouves des solutions équitables et préserves la relation. Sois diplomate et professionnel.`,
      
      loyalty: `Tu es l'IA Fidélisation d'E-Stack AI. Tu gères les programmes fidélité, analas le comportement client et augmentes la lifetime value. Sois orienté expérience client.`,

      // ===== RESSOURCES HUMAINES =====
      hr: `Tu es l'IA RH d'E-Stack AI. Tu gères la stratégie RH, la politique salariale et le développement des talents. Sois empathique et stratégique.`,
      
      recruiter: `Tu es l'IA Recruteur d'E-Stack AI. Tu sourcing les candidats, réalises les entretiens et recommandez les meilleurs profils. Sois perspicace et humain.`,
      
      trainer: `Tu es l'IA Formateur d'E-Stack AI. Tu crées des formations, évalues les compétences et develops les talents. Sois pédagogue et patient.`,
      
      evaluator: `Tu es l'IA Évaluateur d'E-Stack AI. Tu réalises les évaluations de performance, les entretiens annuels et les bilans de compétences. Sois objectif et constructif.`,

      // ===== PRODUCTION =====
      production_manager: `Tu es l'IA Responsable Production d'E-Stack AI. Tu planifies la production, optimises les processus et gères les équipes. Sois organisateur et leader.`,
      
      maintenance: `Tu es l'IA Maintenance d'E-Stack AI. Tu gères la maintenance préventive et curative des équipements. Sois technique et prévoyant.`,
      
      quality: `Tu es l'IA Qualité d'E-Stack AI. Tu garantis les standards qualité, gères les certifications et improves les processus. Sois exigeant et méthodique.`,

      // ===== JURIDIQUE =====
      jurist: `Tu es l'IA Juriste d'E-Stack AI. Tu conseilles sur le droit des affaires, le droit du travail et la conformité. Sois expert juridique et rigoureux.`,
      
      contracts: `Tu es l'IA Contrats d'E-Stack AI. Tu rédiges, analas et négocies les contrats commerciaux. Sois précis et protectionniste.`,
      
      compliance: `Tu es l'IA Conformité d'E-Stack AI. Tu assures la conformité réglementaire (RGPD, ISO, etc.) et gères les risques juridiques. Sois vigilant et proactif.`,

      // ===== ACHATS =====
      buyer: `Tu es l'IA Acheteur d'E-Stack AI. Tu gères les achats, sélectionnes les fournisseurs et optimises les coûts d'approvisionnement. Sois négociateur et analytique.`,
      
      supplier_negotiator: `Tu es l'IA Négociateur Fournisseurs d'E-Stack AI. Tu négocies les contrats fournisseurs, les prix et les conditions. Sois persuasif et stratégique.`,

      // ===== SUPPLY CHAIN =====
      warehouse_manager: `Tu es l'IA Gestionnaire d'Entrepôts d'E-Stack AI. Tu organises le stockage, gères les flux et optimises l'espace. Sois méthodique et organisé.`,
      
      stock_manager: `Tu es l'IA Gestionnaire de Stocks d'E-Stack AI. Tu optimises les niveaux de stock, prévois les besoins et évites les ruptures. Sois prévoyant et analytique.`,
      
      transport_manager: `Tu es l'IA Gestionnaire de Transport d'E-Stack AI. Tu organises la logistique, gère les livraisons et optimises les coûts de transport. Sois opérationnel et rigoureux.`,

      // ===== DATA SCIENCE =====
      analyst: `Tu es l'IA Analyste Data d'E-Stack AI. Tu analas les données, génères des rapports et fournis des insights business. Sois analytique et précis.`,
      
      data_scientist: `Tu es l'IA Data Scientist d'E-Stack AI. Tu construis des modèles ML, fais des prédictions et résous des problèmes complexes avec les données. Sois expert en data science.`,
      
      data_engineer: `Tu es l'IA Data Engineer d'E-Stack AI. Tu construis les pipelines de données, gères les bases de données et assures la qualité des données. Sois technique et rigoureux.`,

      // ===== CYBERSÉCURITÉ =====
      soc: `Tu es l'IA SOC (Security Operations Center) d'E-Stack AI. Tu monitors la sécurité 24/7, détectes les menaces et répond aux incidents. Sois vigilant et réactif.`,
      
      intrusion_detection: `Tu es l'IA Détection d'Intrusion d'E-Stack AI. Tu analas le trafic réseau, détectes les comportements suspects et préviens les attaques. Sois analytique et Security-first.`,
      
      vulnerability_analyst: `Tu es l'IA Analyse de Vulnérabilités d'E-Stack AI. Tu scannes les systèmes, identifies les failles et recommandes les correctifs. Sois méthodique et expert en sécurité.`,

      // ===== GÉNÉRAL =====
      general: `Tu es un assistant IA intelligent faisant partie d'E-Stack AI, le système d'exploitation business intelligent. Tu peux répondre à toutes les questions relatives à la gestion d'entreprise. Réponds de manière helpful, professionnelle et précise.`,
    };

    return prompts[role] || prompts.general;
  }

  async orchestrateTask(task: {
    type: string;
    description: string;
    context?: Record<string, any>;
    agentRole?: string;
  }): Promise<{ response: string; agent: string }> {
    const agentRole = task.agentRole || this.detectAgentFromTask(task.type, task.description);
    const systemPrompt = this.getAgentSystemPrompt(agentRole);
    
    const fullPrompt = task.context 
      ? `${task.description}\n\nContexte: ${JSON.stringify(task.context)}`
      : task.description;

    const response = await this.generateText(fullPrompt, {
      system: systemPrompt,
    });

    return {
      response,
      agent: agentRole,
    };
  }

  private detectAgentFromTask(type: string, description: string): string {
    const text = `${type} ${description}`.toLowerCase();
    
    // Direction Générale
    if (text.includes('pdg') || text.includes('direction') || text.includes('stratégie') || text.includes('vision')) {
      return 'pdg';
    }
    if (text.includes('directeur général') || text.includes('dg ') || text.includes('gestion opérationnelle')) {
      return 'dg';
    }
    if (text.includes('assistant') || text.includes('secrétaire') || text.includes('agenda') || text.includes('réunion')) {
      return 'assistant_exec';
    }
    
    // Finance
    if (text.includes('comptab') || text.includes('écriture') || text.includes('grand livre')) {
      return 'accountant';
    }
    if (text.includes('contrôle') || text.includes('budget') || text.includes('coût') || text.includes('rentabilité')) {
      return 'controller';
    }
    if (text.includes('fiscal') || text.includes('impôt') || text.includes('déclaration') || text.includes('taxe')) {
      return 'fiscalist';
    }
    if (text.includes('analys') || text.includes('financier') || text.includes('investissement') || text.includes('valorisation')) {
      return 'financial_analyst';
    }
    if (text.includes('trésorer') || text.includes('liquidit') || text.includes('placement') || text.includes('change')) {
      return 'treasurer';
    }
    
    // Commerce
    if (text.includes('vente') || text.includes('commercial') || text.includes('closing')) {
      return 'salesperson';
    }
    if (text.includes('négoci') || text.includes('contrat') || text.includes('prix')) {
      return 'negotiator';
    }
    if (text.includes('devis') || text.includes('chiffrage') || text.includes('offre')) {
      return 'quotes_manager';
    }
    if (text.includes('commande') || text.includes('livraison') || text.includes('suivi')) {
      return 'orders_manager';
    }
    
    // Marketing
    if (text.includes('marketing') || text.includes('stratégie')) {
      return 'marketer';
    }
    if (text.includes('communauté') || text.includes('reseau social') || text.includes('community')) {
      return 'community_manager';
    }
    if (text.includes('contenu') || text.includes('article') || text.includes('vidéo') || text.includes('script')) {
      return 'content_creator';
    }
    if (text.includes('publicit') || text.includes('ads') || text.includes('campagne pub')) {
      return 'advertiser';
    }
    if (text.includes('seo') || text.includes('référencement') || text.includes('mots-clés') || text.includes('ranking')) {
      return 'seo';
    }
    if (text.includes('étude') || text.includes('marché') || text.includes('concurrenc') || text.includes('mer')) {
      return 'mer';
    }
    if (text.includes('publicit') || text.includes('slogan') || text.includes('copywriting') || text.includes('cta')) {
      return 'copywriter';
    }
    if (text.includes('email') || text.includes('newsletter') || text.includes('automation')) {
      return 'email_marketer';
    }
    if (text.includes('growth') || text.includes('croissance') || text.includes('acquisition')) {
      return 'growth_hacker';
    }
    
    // Service Client
    if (text.includes('chatbot') || text.includes('bot') || text.includes('question')) {
      return 'chatbot';
    }
    if (text.includes('sav') || text.includes('retour') || text.includes('échange')) {
      return 'sav';
    }
    if (text.includes('réclam') || text.includes('plainte') || text.includes('conflict')) {
      return 'claims_manager';
    }
    if (text.includes('fidélit') || text.includes('programme') || text.includes('crm')) {
      return 'loyalty';
    }
    
    // RH
    if (text.includes('rh') || text.includes('ressource humaine') || text.includes('talent')) {
      return 'hr';
    }
    if (text.includes('recrut') || text.includes('candidat') || text.includes('sourcing') || text.includes('entretien')) {
      return 'recruiter';
    }
    if (text.includes('formation') || text.includes('former') || text.includes('compétence')) {
      return 'trainer';
    }
    if (text.includes('évalu') || text.includes('performance') || text.includes('bilan')) {
      return 'evaluator';
    }
    
    // Production
    if (text.includes('production') || text.includes('planification') || text.includes('fabrication')) {
      return 'production_manager';
    }
    if (text.includes('maintenance') || text.includes('équipement') || text.includes('dépannage')) {
      return 'maintenance';
    }
    if (text.includes('qualit') || text.includes('certification') || text.includes('iso')) {
      return 'quality';
    }
    
    // Juridique
    if (text.includes('juridi') || text.includes('droit') || text.includes('légal')) {
      return 'jurist';
    }
    if (text.includes('contrat') || text.includes('avenant') || text.includes('clause')) {
      return 'contracts';
    }
    if (text.includes('conformit') || text.includes('rgpd') || text.includes('règlement')) {
      return 'compliance';
    }
    
    // Achats
    if (text.includes('achat') || text.includes('approvisionnement') || text.includes('fournisseur')) {
      return 'buyer';
    }
    if (text.includes('fournisseur') || text.includes('négociation') || text.includes('achat')) {
      return 'supplier_negotiator';
    }
    
    // Supply Chain
    if (text.includes('entrepôt') || text.includes('stockage') || text.includes('magasin')) {
      return 'warehouse_manager';
    }
    if (text.includes('stock') || text.includes('inventaire') || text.includes('rupture')) {
      return 'stock_manager';
    }
    if (text.includes('transport') || text.includes('logistique') || text.includes('livraison')) {
      return 'transport_manager';
    }
    
    // Data Science
    if (text.includes('analys') || text.includes('donnee') || text.includes('rapport') || text.includes('kpi')) {
      return 'analyst';
    }
    if (text.includes('machine learning') || text.includes('modèle') || text.includes('prédiction') || text.includes('ml')) {
      return 'data_scientist';
    }
    if (text.includes('pipeline') || text.includes('base de données') || text.includes('etl')) {
      return 'data_engineer';
    }
    
    // Cybersécurité
    if (text.includes('sécurit') || text.includes('monitor') || text.includes('incident')) {
      return 'soc';
    }
    if (text.includes('intrusion') || text.includes('attaqu') || text.includes('malware')) {
      return 'intrusion_detection';
    }
    if (text.includes('vulnérabilit') || text.includes('failles') || text.includes('pentest')) {
      return 'vulnerability_analyst';
    }
    
    return 'general';
  }
}
